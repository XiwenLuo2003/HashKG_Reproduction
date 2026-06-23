import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from argparse import ArgumentParser
import torch

from netquery.utils import *
from netquery.data_utils import load_test_queries_by_formula
from netquery.model import QueryEncoderDecoder
from netquery.hashed_model import HashedQueryEncoderDecoder
from netquery.train_helpers import run_eval
from netquery.fb.data_utils import load_graph_and_ontology
from netquery.fb.encoders import WHEEncoder, RHEEncoder

parser = ArgumentParser()
parser.add_argument("--embed_dim", type=int, default=128)
parser.add_argument("--data_dir", type=str, default="../../fb15k_data")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--beta", type=int, default=20)
parser.add_argument("--log_dir", type=str, default="./log")
parser.add_argument("--model_dir", type=str, default="./model")
parser.add_argument("--decoder", type=str, default="bilinear")
parser.add_argument("--inter_decoder", type=str, default="mean")
parser.add_argument("--type_encoder", type=str, default="whe", choices=["whe", "rhe", "direct"])
parser.add_argument("--max_forward_size", type=int, default=2048,
                    help="Max nodes per forward pass during percentile eval (lower if OOM)")

args = parser.parse_args()

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

print("Loading graph data and ontology...")
graph, feature_modules, node_maps, ontology_data = load_graph_and_ontology(args.data_dir, args.embed_dim)

# Apply cudify manually for evaluation if needed
# We evaluate on CPU or GPU. Let's use GPU if available.
cuda = torch.cuda.is_available()
if cuda:
    graph.features = cudify(feature_modules, node_maps)
    ontology_data['entity_types'] = ontology_data['entity_types'].cuda()
    ontology_data['entity_weights'] = ontology_data['entity_weights'].cuda()
    ontology_data['entity_domains'] = ontology_data['entity_domains'].cuda()

out_dims = {mode: args.embed_dim for mode in graph.relations}

print("Loading query data..")
test_queries = load_test_queries_by_formula(args.data_dir + "/test_edges.pkl")
for i in range(2, 4):
    i_test_queries = load_test_queries_by_formula(args.data_dir + f"/test_queries_{i}.pkl")
    test_queries["one_neg"].update(i_test_queries["one_neg"])
    test_queries["full_neg"].update(i_test_queries["full_neg"])

print(f"Initializing {args.type_encoder.upper()} Encoder...")
if args.type_encoder == "whe":
    enc = WHEEncoder(graph.features, feature_modules, ontology_data, args.embed_dim, cuda, beta=args.beta)
elif args.type_encoder == "rhe":
    enc = RHEEncoder(graph.features, feature_modules, ontology_data, args.embed_dim, cuda, beta=args.beta)
else:
    from netquery.encoders import DirectEncoder
    enc = DirectEncoder(graph.features, feature_modules, beta=args.beta)

dec = get_metapath_decoder(graph, out_dims, args.decoder, beta=args.beta)
inter_dec = get_intersection_decoder(graph, out_dims, args.inter_decoder, beta=args.beta)

print('Loading model...')
# Load the fine-tuned model (not edge_conv) if available, otherwise edge_conv
model_file = args.model_dir + f"/{args.data_dir.strip().split('/')[-1]}-{args.type_encoder}-{args.beta}-128-{args.lr:.6f}-{args.decoder}-{args.inter_decoder}"
if not os.path.exists(model_file):
    model_file = model_file + "-edge_conv"

if os.path.exists(model_file):
    state_dict = torch.load(model_file)
else:
    raise FileNotFoundError(f"Model file {model_file} not found!")

enc_dec = QueryEncoderDecoder(graph, enc, dec, inter_dec)
enc_dec.load_state_dict(state_dict)

hashed_enc_dec = HashedQueryEncoderDecoder(graph, enc, dec, inter_dec)
hashed_enc_dec.load_state_dict(state_dict)

if cuda:
    enc_dec.cuda()
    hashed_enc_dec.cuda()

log_file = args.log_dir + f"/{args.type_encoder}_test_result.log"
logger = setup_logging(log_file)

logger.info(f"Testing original continuous model with {args.type_encoder.upper()}...")
run_eval(enc_dec, test_queries, 0, logger, max_forward_size=args.max_forward_size)

logger.info(f"Testing HASHED model with {args.type_encoder.upper()}...")
run_eval(hashed_enc_dec, test_queries, 0, logger, max_forward_size=args.max_forward_size)
