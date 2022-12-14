# to profile:
#
#  export CPROFILE=1
# run as usual
#
# python scripts/prstats.py RUNDIR/ITER/*/*stats > pr.txt
# look at pr.txt

import pstats, sys
p = pstats.Stats(*sys.argv[1:])
#p.strip_dirs().sort_stats(-1).print_stats()
p.sort_stats('cumulative').print_stats()
#p.sort_stats('calls').print_stats(1000)

# for >3.6
#from pstats import SortKey
#p = pstats.Stats('profout.stats')
#p.strip_dirs().sort_stats(-1).print_stats()
