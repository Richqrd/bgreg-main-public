from bgreg.data.analysis.graph.patch import patch


gr_dir = "jh101_grs.pickle"
logdir = "test_gr_patch"
file_name = "jh101_grs"
patch(gr_dir=gr_dir, logdir=logdir, file_name=file_name, mode="binary", model="supervised", stats=True)
