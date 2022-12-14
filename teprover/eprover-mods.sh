# No one should ever need to run this script while I (VRA) am still working on this project.
# A compiled version of eprover is included with Trail.

# This file patches eprover/PROVER/eprover.c for Trail.
# Use the output of this file to replace eprover/PROVER/eprover.c.

# The last two lines of eprover.c:main() should be:
#   OutClose(GlobalOut);
#   return retval;

# (base) Vernons-MBP:Trail austel$ scp trail-eprover/modified2.6/cco_proofproc.c trail10.sl.cloud9.ibm.com:eprover/CONTROL/cco_proofproc.c ; scp trail-eprover/modified2.6/eprover.c trail10.sl.cloud9.ibm.com:eprover/PROVER
#(trail) austel@trail10:/home/austel/Trail$ make -C ~/eprover; cp ~/eprover/PROVER/eprover ~/Trail/trail-eprover/bin/eprover.p1

#cnf(i_0_25, plain, (X1=k1_xboole_0|~v1_xboole_0(X1))).
#cnf(i_0_25, plain, (X1=o_0_0_xboole_0|~v1_xboole_0(X1))).

# just for uniformity, use the same args as the other script

# NOW VERSIONING
# I now have original/modified dirs for eprover versions; currently just 2.6
# The particular snapshot of 2.6 is in Trail branch 'eprover_2.6'.
# The files in the 'original2.6' dir are just copied from that branch.

VERSION=${1:?you must specify an eprover version}
P=${2:?you must supply the eprover protocol version as an argument} 

O=original$VERSION
M=modified$VERSION

FILES="cco_proofproc.c cco_scheduling.c eprover.c"

if ! [[ -d $O ]]; then
    echo "We don't have that version!"
    echo "copy $FILES of that eprover version to $O"
    exit 1;
fi

mkdir -p $M

cd $O
if ! ls $FILES &>/dev/null; then
    echo "missing some files;need $FILES"
    exit 1;
fi
cd ..

perl -n -e '

if ($_ eq "int main(int argc, char* argv[])\n") {
   print("  extern void te_get_envvars(void);\n");
   $pr=1;
   print;
} else {
  if (/Failure: User resource limit exceeded/) {
  # nuts
  }  else {
    print;
  }
  if ($pr) {
      print("  te_get_envvars();\n");
      $pr=0;
  }
}

if (/OutClose\(GlobalOut\)/) {
   print("   extern void writeDummyTrailInput();\n");
   print("   writeDummyTrailInput(); // this lets know that the proof has been written; avoids an error on read.\n");
}

if (/Failure: User resource limit exceeded/) {
   print("         extern int trail_stopping;\n");
#   print("         printf(\"trail_stopping: %d\\n\", trail_stopping);\n");
   print("         if (trail_stopping) {\n");
   print("           PrintProofObject = 0;\n");
   print("         } else {\n");
   $ro=1;
   # there are two cases of "retval = RESOURCE_OUT"
}
if ($ro && /retval = RESOURCE_OUT/) {
   print("         }\n");
   $_="";
}
' $O/eprover.c > $M/eprover.c

if ! grep writeDummyTrail $M/eprover.c &>/dev/null; then
    echo "The patch failed! Sorry"
    exit 1
fi
echo eprover.c successfully patched.


# in ExecuteScheduleMultiCore
# hmm
# in ExecuteSchedule
{
    s='ScheduleTimesInitMultiCore(strats, run_time, 4);'
    s='ScheduleTimesInitMultiCore(strats, run_time, cores);'
#    s='pid_t ExecuteSchedule(ScheduleCell strats[],'

    fgrep -B 10000 "$s" $O/cco_scheduling.c | head -n -1

cat <<EOF    
  char *tauto=getenv("TRAIL_AUTO");
  if (tauto && strlen(tauto)) {
      int i=atoi(tauto);
      h_parms->heuristic_name         = strats[i].heu_name;
      h_parms->order_params.ordertype = strats[i].ordering;
      strats[i].time_absolute = ScheduleTimeLimit-run_time;  // instead of ScheduleTimesInit
      SetSoftRlimit(RLIMIT_CPU, strats[i].time_absolute); // as in Child branch      
      printf("Trail using auto strategy %d\n", i);
      rlim_t tmp=ScheduleTimeLimit-run_time;
      fprintf(GlobalOut,
              "# %s assigned %ju seconds\n",
              strats[i].heu_name, (uintmax_t)tmp);
      return;
  } else 
     printf("TRAIL_AUTO not set %p\n",tauto);
EOF

    fgrep -A 10000 "$s" $O/cco_scheduling.c
} > $M/cco_scheduling.c

if ! grep TRAIL_AUTO $M/cco_scheduling.c &>/dev/null; then
    echo "The patch failed for cco_scheduling.c! Sorry"
    exit 1
fi
echo cco_scheduling.c successfully patched.




# No one should ever need to run this script while I (VRA) am still working on this project.
# A compiled version of eprover is included with Trail.

# This script patches Trail code into modified/cco_proofproc.c file.
# It assumes that the original eprover version of the code is in the cwd.
# The changes are:
# a couple of header files
# code to set 'clause' to 'trailClause' in ProcessClause()
# code to call communication code in Saturate()
# after that, all the new Trail code, which is in file cco_proofproc_trail.c.
# Note that that file is NOT to be compiled itself!
# It is just more convenient to keep it in a separate file.
# It is cat'ed at the end of this file.
# Use the output of this file to replace cco_proofproc.c.

(
cat <<EOF
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <sys/file.h>
EOF

perl -n -e '
if (/^      assert\(clause->neg_lit_no == 0\);/) {
   printf("      extern int processed_demodulator;\n");
   printf("      processed_demodulator=1;\n");
}

if (/^Clause_p ProcessClause/) {
  $s = <<'HEREDOC';
static Clause_p trailClause; // HACK - avoid hassle adding param
static void writeTrailInput(ProofState_p state, ProofControl_p control);
static void maybeOpenTrailPipes(FILE **trailInputp, FILE **trailOutputp);
static Clause_p readTrailOutput(ProofState_p state, FILE *trailInput, FILE *trailOutput);
static void reopenTrailPipes(FILE *trailInput, FILE *trailOutput);
static Clause_p maybe_select_single_lit(ProofState_p state);
static int always_select_single_lit_action=0;
HEREDOC

  print $s;
}

if (/clause = control->hcb->hcb_select/) {
  print("   if (trailClause != NULL) {\n");
  print("      clause = trailClause;\n");
  print("   } else\n");
}

# while-loop condition of Saturate()
if (/while\(!TimeIsUp/) {
  $s = <<'HEREDOC';

   // START TRAIL MODS
   static FILE *trailInput=NULL, *trailOutput = NULL;
   // there is no actual flag for this.
   int presaturating = (control->heuristic_parms.selection_strategy == SelectNoGeneration &&
                        step_limit == LONG_MAX &&
                        proc_limit == LONG_MAX &&
                        unproc_limit == LONG_MAX &&
                        total_limit == LONG_MAX &&
                        generated_limit == LONG_MAX &&
                        tb_insert_limit == LONG_MAX);

   static int trail_inited=0;
   if (!trail_inited) {
     trail_inited=1;

     if (!trailInput) 
        maybeOpenTrailPipes(&trailInput, &trailOutput);
   }
   char *skip_pre = getenv("SKIP_PRESATURATE");
   if (presaturating) {
     if (skip_pre && strlen(skip_pre)) {
       printf("Saturate - presumably presaturating - skipping since SKIP_PRESATURATE\\n");
       return 0;
     }
   }
//     fprintf(trailInput, "%d\\n", presaturating);
//     fflush(trailInput);
   if (presaturating) {
       // untested, never seen so far
       printf("Saturate - presumably presaturating\\n");
   } else
       printf("Saturate - presumably NOT presaturating\\n");
   // END TRAIL MODS
HEREDOC

  print $s;
}

# inside while-loop of Saturate()
if (/unsatisfiable = ProcessClause/) {
  $s = <<'HEREDOC';
       if (trailInput) {
         trailClause=0;
         if (always_select_single_lit_action)
           trailClause=maybe_select_single_lit(state);
         if (!trailClause) {
             writeTrailInput(state,control);

         // THIS COMMENT IS OUT OF DATE but may still be helpful.
       // we hang here waiting for Trail to execute sendAction(clause)
       // this is confusing.
       // eprover                     Trail
       //    writeTrailInput          
       //                          init_game ->  readStateStrings
       //                          while 1  
       //                               sendAction
       //    readTrailOutput    
       // while 1                  
       //    write +/-MAGIC           
       //                               done?
       //    writeTrailInput          
       //                               nextStep -> readStateStrings
       //    readTrailOutput
       //    
       // The two loops are shifted with respect to each other.
       // we pass trailInput only because if we fork we need to reopen it
          trailClause = readTrailOutput(state, trailInput, trailOutput);
          if (!trailClause)
             break; // STOP - presumably actually already done
        }
      }
HEREDOC

  print $s;
}


print;
' $O/cco_proofproc.c

echo "static int EPROVER_PROTOCOL=$P;"

cat cco_proofproc_trail.c
) > $M/cco_proofproc.c

if fgrep '     maybeOpenTrailPipes(' $M/cco_proofproc.c &>/dev/null &&
        fgrep '       if (trailInput) {' $M/cco_proofproc.c &>/dev/null; then
    echo cco_proofproc.c sucessfully patched
else
    echo cco_proofproc.c patch failed
    exit 1
fi


