# https://docs.python.org/3.6/library/profile.html

# To use this, edit the last line of launch-ee.sh, then run this.
import pstats
p = pstats.Stats('profout.stats')
#p.strip_dirs().sort_stats(-1).print_stats()
p.sort_stats('cumulative').print_stats(100)
