from projects.gnn_to_soz.main import run

a = "corr"
nf = "ones"
ef = "ones"
run("jh101", node_feat=nf, edge_feat=ef, adj=a, normalize=True,
    link_cutoff=0.3, classify="binary", combined_features=False)
