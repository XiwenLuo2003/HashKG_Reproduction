import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from argparse import ArgumentParser
import torch
from torch import optim

from netquery.utils import *
from netquery.data_utils import load_test_queries_by_formula, load_queries_by_formula
from netquery.model import QueryEncoderDecoder
from netquery.train_helpers import run_train
from netquery.fb.data_utils import load_graph_and_ontology
from netquery.fb.encoders import WHEEncoder, RHEEncoder

parser = ArgumentParser()
parser.add_argument("--embed_dim", type=int, default=128)
parser.add_argument("--data_dir", type=str, default="../../fb15k_data")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--max_iter", type=int, default=100000000)
parser.add_argument("--max_burn_in", type=int, default=1000000)
parser.add_argument("--val_every", type=int, default=5000)
parser.add_argument("--tol", type=float, default=0.0001)
parser.add_argument("--beta", type=int, default=1)
parser.add_argument("--cuda", action='store_true', default=True)
parser.add_argument("--log_dir", type=str, default="./log")
parser.add_argument("--model_dir", type=str, default="./model")
parser.add_argument("--decoder", type=str, default="bilinear")
parser.add_argument("--inter_decoder", type=str, default="mean")
parser.add_argument("--opt", type=str, default="adam")
parser.add_argument("--pretrain", type=bool, default=False)
parser.add_argument("--type_encoder", type=str, default="whe", choices=["whe", "rhe", "direct"], 
                    help="Choose the entity encoder type: 'whe', 'rhe', or 'direct'.")

args = parser.parse_args()

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)

print("Loading graph data and ontology...")
graph, feature_modules, node_maps, ontology_data = load_graph_and_ontology(args.data_dir, args.embed_dim)

if args.cuda:
    graph.features = cudify(feature_modules, node_maps)
    ontology_data['entity_types'] = ontology_data['entity_types'].cuda()
    ontology_data['entity_weights'] = ontology_data['entity_weights'].cuda()
    ontology_data['entity_domains'] = ontology_data['entity_domains'].cuda()

out_dims = {mode: args.embed_dim for mode in graph.relations}

print("Loading edge data..")
train_queries = load_queries_by_formula(args.data_dir + "/train_edges.pkl")
val_queries = load_test_queries_by_formula(args.data_dir + "/val_edges.pkl")
test_queries = load_test_queries_by_formula(args.data_dir + "/test_edges.pkl")

if not args.pretrain:
    print("Loading complex query data..")
    for i in range(2, 4):
        train_queries.update(load_queries_by_formula(args.data_dir + f"/train_queries_{i}.pkl"))
        i_val_queries = load_test_queries_by_formula(args.data_dir + f"/val_queries_{i}.pkl")
        val_queries["one_neg"].update(i_val_queries["one_neg"])
        val_queries["full_neg"].update(i_val_queries["full_neg"])
        i_test_queries = load_test_queries_by_formula(args.data_dir + f"/test_queries_{i}.pkl")
        test_queries["one_neg"].update(i_test_queries["one_neg"])
        test_queries["full_neg"].update(i_test_queries["full_neg"])

print(f"Initializing {args.type_encoder.upper()} Encoder...")
if args.type_encoder == "whe":
    enc = WHEEncoder(graph.features, feature_modules, ontology_data, args.embed_dim, args.cuda, beta=args.beta)
elif args.type_encoder == "rhe":
    enc = RHEEncoder(graph.features, feature_modules, ontology_data, args.embed_dim, args.cuda, beta=args.beta)
else:
    from netquery.encoders import DirectEncoder
    enc = DirectEncoder(graph.features, feature_modules, beta=args.beta)

dec = get_metapath_decoder(graph, out_dims, args.decoder, args.beta)
inter_dec = get_intersection_decoder(graph, out_dims, args.inter_decoder, args.beta)

enc_dec = QueryEncoderDecoder(graph, enc, dec, inter_dec)
if args.cuda:
    enc_dec.cuda()

if args.beta != 1:
    last_model_file = args.model_dir + f"/{args.data_dir.strip().split('/')[-1]}-{args.type_encoder}-{args.beta-1}-128-{args.lr:.6f}-{args.decoder}-{args.inter_decoder}-edge_conv"
    print(f"Loading previous KG embedding when beta={args.beta-1} from {last_model_file}")
    if os.path.exists(last_model_file):
        enc_dec.load_state_dict(torch.load(last_model_file))
    else:
        print(f"WARNING: Previous model file {last_model_file} not found. Continuing from scratch.")

if args.opt == "sgd":
    optimizer = optim.SGD([p for p in enc_dec.parameters() if p.requires_grad], lr=args.lr, momentum=0)
elif args.opt == "adam":
    optimizer = optim.Adam([p for p in enc_dec.parameters() if p.requires_grad], lr=args.lr)
    
log_file = args.log_dir + f"/{args.data_dir.strip().split('/')[-1]}-{args.type_encoder}-{args.beta}-128-{args.lr:.6f}-{args.decoder}-{args.inter_decoder}.log"
model_file = args.model_dir + f"/{args.data_dir.strip().split('/')[-1]}-{args.type_encoder}-{args.beta}-128-{args.lr:.6f}-{args.decoder}-{args.inter_decoder}"
logger = setup_logging(log_file)

run_train(enc_dec, optimizer, train_queries, val_queries, test_queries, logger, 
          max_burn_in=args.max_burn_in, val_every=args.val_every, model_file=model_file, pretrain=args.pretrain)

if not args.pretrain:
    torch.save(enc_dec.state_dict(), model_file)
