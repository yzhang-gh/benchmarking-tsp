# Rscript call_netgen.R point_num clu.lower clu.upper ins_num seed data_dir
suppressPackageStartupMessages(library("netgen"))
suppressPackageStartupMessages(library("ggplot2"))
args <- commandArgs(TRUE)
points.num <- as.integer(args[1])
clu.lower <- as.integer(args[2])
clu.upper <- as.integer(args[3])
ins.num <- as.integer(args[4])
seed <- as.integer(args[5])
data_dir <- args[6]

set.seed(seed)

index = 0
for (clu_num in clu.lower:clu.upper)
{
    for (i in 1:ins.num)
    {
        x = generateClusteredNetwork(n.cluster = clu_num, n.points = points.num)
        # by default, lower = 0, upper = 100, out.of.bounds.handling = "mirror"
        x = rescaleNetwork(x)
        # by default, method = "global2"
        # portgen range 1~1000000 https://www.cs.uwyo.edu/~larsko/papers/kotthoff_improving_2015.pdf
        x$coordinates = x$coordinates * 1000000
        x$coordinates[x$coordinates == 0] <- 0.1
        x$coordinates = ceiling(x$coordinates)
        x$lower = 1
        x$upper = 1000000
        filepath = sprintf("%s/clust%d_seed%d_%d.tsp", data_dir, points.num, seed, index)
        exportToTSPlibFormat(x, filepath, use.extended.format=FALSE)
        index = index + 1
    }
}
