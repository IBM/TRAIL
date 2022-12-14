// This file is (of course) NOT compiled independently;
// it is used by patch-eprover.sh to create a version of cco_proofproc.c for use with Trail.
// If you change this file, you must:
//    run patch-eprover.sh to replace cco_proofproc.c on your installation of eprover
//    recompile eprover
//    copy the new version of eprover to the Trail dir (and check it in!).
#include <signal.h>
#include <sys/prctl.h>

static FILE *trailInputX;
static FILE *trailOutputX;
static FILE *trailInputBinX;
static int check_selected_clause = 0;
static FILE *selectedClauseFile;
static int STOP_COMMAND; // assigned on startup, not changed
static int FORK_COMMAND; // assigned on startup, not changed

static void maybeOpenTrailPipes(FILE **trailInputp, FILE **trailOutputp) {
   char *trailInputName = "E2Trail.pipe";
   char *trailInputBinName = "E2Trail.pipeBin";
   char *trailOutputName = "Trail2E.pipe";
   int trailInputExists  = (access(trailInputName,  F_OK ) == 0);
   int trailOutputExists = (access(trailOutputName, F_OK ) == 0);
   //   int trailInputBinExists = (access(trailInputBinName, F_OK ) == 0);
   int followTrail = (trailInputExists || trailOutputExists);

   FILE *trailInput = NULL, *trailOutput = NULL;

   if (followTrail) {
     // we need this for Beam Search, otherwise we get zombie procs
     // however, it somehow causes eprover to hang when run in non-Trial mode,
     // so we only do it in that case.
     if (!getenv("MEASURE_FORK_OVERHEAD") && !getenv("MEASURE_JUST_EPROVER_FORK_OVERHEAD"))
       // I think wait() won't return the child pid properly if we do this;
       // not necessary, but I like that as a check
       signal(SIGCHLD, SIG_IGN); /* Silently (and portably) reap children. */

     prctl(PR_SET_PDEATHSIG, SIGKILL, 0, 0, 0); // die when parent (bash shell) dies
     
     if (!(trailInputExists && trailOutputExists)) {
       // I want to avoid any possible problems by opening with "r"
       printf("argument for trail-input doesn't exist: '%s'\n", trailInputName);
       exit(1);
     }

     printf("TRAIL INPUT EXISTS2: %s\n", trailInputName);
     if ((trailInput = fopen(trailInputName, "w")) == NULL) {
       printf("argument for trail-input isn't writable: '%s'\n", trailInputName);
       exit(1);
     }
     printf("eprover Opened trailInput\n");
     fflush(stdout);
     trailInputX = trailInput;
     fprintf(trailInput, "eproverTrail protocol %d\n", EPROVER_PROTOCOL); // see e_prover.py where this line is read
     printf("eprover wrote first dummy line\n");
     fflush(trailInput);

     printf("eprover opening trailOutput\n");
     // we read Trail's output
     if((trailOutput = fopen(trailOutputName, "r")) == NULL) {
       printf("trail-output doesn't exist or isn't readable: '%s'\n", trailOutputName);
       exit(1);
     }
     printf("eprover Opened trailOutput\n");
#if 0
     /* setvbuf(trailOutput, NULL, _IOLBF, 100); // need this for forked children, since we share the input */
     setvbuf(trailOutput, NULL, _IONBF, 100); // need this for forked children, since we share the input
     /* setbuffer(trailOutput, 0, 0); */
#endif
     
     if (fscanf(trailOutput, "%d", &STOP_COMMAND)!=1) {
       printf("Reading first line from trail failed!\n");
       exit(1);
     }
     printf("eprover read STOP_COMMAND from trailOutput: %d\n", STOP_COMMAND);
     if (fscanf(trailOutput, "%d", &FORK_COMMAND)!=1) {
       printf("Reading second line from trail failed!\n");
       exit(1);
     }
     printf("eprover read FORK_COMMAND from trailOutput: %d\n", FORK_COMMAND);

     if ((trailInputBinX = fopen(trailInputBinName, "wb")) == NULL) {
       printf("argument for trail-input isn't writable: '%s'\n", "processed");
       exit(1);
     }
     fwrite(&STOP_COMMAND, sizeof(int), 1, trailInputBinX);
     fwrite(&FORK_COMMAND, sizeof(int), 1, trailInputBinX);
     fflush(trailInputBinX);
     printf("eprover wrote %d and %d to trailInputBin\n", STOP_COMMAND, FORK_COMMAND);
     fflush(stdout);

     if (check_selected_clause) {
       printf("opening \"selected_clause.txt\"\n");
       if (! (selectedClauseFile = fopen("selected_clause.txt", "w"))) {
         printf("Can't open \"selected_clause.txt\"");
         exit(1);
       }
     }
   }

   *trailInputp = trailInput;
   *trailOutputp = trailOutput;
   trailOutputX = trailOutput;
}

// nuts
static void reopenTrailPipes(FILE *trailInput, FILE *trailOutput) {
   char *trailInputName = "E2Trail.pipe";
   char *trailOutputName = "Trail2E.pipe";
   char *trailInputBinName = "E2Trail.pipeBin";
   
   printf("eprover freopening trailOutput\n");

   if(freopen(trailOutputName, "r", trailOutput) == NULL) {
     printf("trail-output doesn't exist or isn't readable: '%s'\n", trailOutputName);
     exit(1);
   }
   printf("eprover Opened trailOutput\n"); fflush(stdout);

   char buf[128];
   if (fscanf(trailOutput, "%s", buf)!=1) {
       printf("Reading first line from trail failed!\n");
       exit(1);
     }
   printf("read %s from Trail\n", buf);  fflush(stdout);
   trailOutputX = trailOutput;
   
   // at one point, the child eprover process sometimes just stopped after the 'reopening' message above.
   // as a result, e_prover.py failed when trying to open the eprover /proc dir (FileNotFoundError).
   // it doesn't seem to do that with in the other version (fclose/fopen).
   // I have no idea why.
   // hmmm... it turned out I did in fact still get errors, and there is the hassle of the pointer changing.
   // now trying a hack in e_prover.py to just try again.
   // hmmm... I think it was just because the prover was running out of (real) time (not using cpu time).
   // I wasn't reopening stderr.
   if (freopen(trailInputName, "w", trailInput) == NULL) {
     printf("argument for trail-input isn't writable: '%s'\n", trailInputName); fflush(stdout);
     exit(1);
   }
   printf("eprover reopened trailInput\n"); fflush(stdout);
   trailInputX = trailInput;
   
   fprintf(trailInput, "%d\n", getpid());    fflush(trailInput);
   printf("eprover wrote pid\n"); fflush(stdout);


   if (freopen(trailInputBinName, "wb", trailInputBinX) == NULL) { 
     printf("argument for trail-inputbin isn't writable: '%s'\n", trailInputBinName); fflush(stdout);
     exit(1);
   }
   printf("eprover reopened trailInputBin\n"); fflush(stdout);
   
   int pid=getpid();
   fwrite(&pid, sizeof(int), 1, trailInputBinX);    fflush(trailInputBinX);
   printf("eprover wrote pid\n"); fflush(stdout);   
   

   /* setvbuf(trailOutput, NULL, _IONBF, 100); // need this for forked children, since we share the input */
}

#if 0
#define CLAUSESTRBUF_SZ (10*1000*1000)
#define OURCLAUSES_SZ (100*1000)
static size_t nclauseStrBuf;
static char clauseStrBuf[OURCLAUSEBUF_SZ];
static size_t nclauseStrs=1;

static size_t nclauseStrIndex;
static int clauseStrIndex[OURCLAUSES_SZ]; 
#endif

#define MASKTP long


// inited to 0 since global
typedef struct EpInfo {
  //  int firstAgeProcessed;
  //  int firstAgeUnprocessed;
  int uclause;
  char selected_literal_inited;
  char processed; // processed vs avaiable
  MASKTP selected_literal;
  // haven't found any field that indicates rewriting
  //PStack_p derivation; // I believe that if the clause is rewritten, then this must change. WRONG
  //long create_date; // unchanging
  //long proof_size;  // unchanging?
  char *clauseStr;
} *EpInfo_t;

#define UCLAUSE_SZ (1000*1000)
//char countBufs[2][2][UCLAUSE_SZ];
char countBufs[UCLAUSE_SZ];
int nclauseids;

#define EPIDS_SZ (1000*1000)
static size_t nepids;
static struct EpInfo epids[EPIDS_SZ];


#if 0
static void writeClauseSetIds2Trail(FILE *trailInput, ClauseSet_p clauseset)
{
  for(Clause_p handle = clauseset->anchor->succ; 
      handle!=clauseset->anchor; handle = handle->succ) {
    int epid = handle->ident-LONG_MIN; 
    printf("%d ", epid);
  }
}
#endif

#if 0
static void writeClauseSet(ClauseSet_p clauseset)
{
  for(Clause_p handle = clauseset->anchor->succ; 
      handle!=clauseset->anchor; handle = handle->succ) {
    int epid = handle->ident-LONG_MIN; 
    printf("%d ", epid);
    ClauseTSTPPrint(trailInput, handle, true, true);
    printf("\n");
static char *clauseStrs[OURCLAUSES_SZ];  }
}
#endif


static Clause_p getclause(ClauseSet_p clauses, int idx)
{
  int i = 0;
  Clause_p ret = NULL;
  Clause_p handle;
  for(handle = clauses->anchor->succ;
      handle != clauses->anchor && i < idx;
      handle = handle->succ) {
    i++;
  }
  if (i == idx && handle != clauses->anchor) {
    ret = handle;
  }
  return ret;
#if 0
  for(Clause_p handle = clauseset->anchor->succ; handle!=clauseset->anchor; handle = handle->succ) {
    int epid = handle->ident-LONG_MIN;
    if (epids[epid].uclause == idx)
      return handle;
  }
  printf("couldn't find idx %d\n",idx);
  exit(1);
#endif
}

static int dbg_sellit = 0;
static int print_selected_clause = 0;
//static int check_selected_clause = 0;
//static int always_select_single_lit_action=0;

//static void clause_select_pos(Clause_p clause)
static MASKTP clause_get_selected(Clause_p clause)
{
   Eqn_p handle=clause->literals;

   //assert(clause->neg_lit_no);
   int i=0;
   MASKTP sel=0;
   while(handle)
   {
     if(EqnIsSelected(handle))
      {
        if (i>=8*sizeof(long)) {
          printf("LONG SEL LIT! %d\n", i);
          break;
        }
        sel |= (1<<i);
        if (dbg_sellit) 
          printf("%d ", i);
      }
      handle = handle->next;
      i++;
   }
   if (dbg_sellit) {
   printf("CGS %d %ld ", i, sel);
   ClauseTSTPPrint(stdout, clause, true, true);
   printf("\n");
   }
   
   if (dbg_sellit && !sel && i>1) {
     printf("NO SELECTION! ");
     ClauseTSTPPrint(stdout, clause, true, true);
     printf("\n");
   }
   return sel;
}

int processed_demodulator;

static int get_literal_selection = -1;
static int register_on_processed;
static int register_on_demodulator;
static int register_on_change;
static int register_every_nsteps;
static int save_clause_str_in_mem;
static int re_register_on_single_lit_action;
static int dropdups=0;
//static int dbg_sellit = 0;
// this is a mess
void te_get_envvars() {
  char *s = getenv("TE_GET_LITERAL_SELECTION");

  get_literal_selection = s && s[0]; // set and non-empty
  printf("te opts: get_literal_selection=%d ", get_literal_selection);

#if 0
  // obsolete
  s = getenv("TE_REGISTER_EVERY_NSTEPS");
  if (s[0])
    register_every_nsteps=atoi(s);
  
  s = getenv("TE_REGISTER");
#define TE_GETOPT(nm) if (strstr(s, #nm)) { nm=1; printf(" %s=%d", #nm, nm); }

  // obsolete
  TE_GETOPT(register_on_processed);
  TE_GETOPT(register_on_demodulator);
  TE_GETOPT(register_on_change);

  //TE_GETOPT(save_clause_str_in_mem);
  if (register_on_demodulator || register_on_change) 
    save_clause_str_in_mem=1;
  
  TE_GETOPT(dbg_sellit);
#endif

  print_selected_clause = getenv("PRINT_SELECTED_CLAUSE")[0];
  check_selected_clause = getenv("CHECK_SELECTED_CLAUSE")[0];
  re_register_on_single_lit_action = getenv("RE_REGISTER_ON_SINGLE_LIT_ACTION")[0];

  always_select_single_lit_action=getenv("ALWAYS_SELECT_SINGLE_LIT_ACTION")[0];

  dropdups=getenv("DROPDUPS")[0]-'0';
  printf("dd %d\n", dropdups);
  printf(" print_selected_clause=%d\n", print_selected_clause);
  fflush(stdout);
  //TE_GETOPT(get_literal_selection);
}

// called from DoLiteralSelection
void trail_selected(Clause_p clause) {
  if (!get_literal_selection)
    return;
  
  int epid = clause->ident-LONG_MIN;
  struct EpInfo *ep = &epids[epid];

  MASKTP sel = clause_get_selected(clause);
  if (dbg_sellit) {
    printf("trail selected literal: %d %ld ", epid, sel
#if 0
         !ep->proof_size ? "" :
         ep->proof_size == clause->proof_size ? "SMX" : "DFX"
         !ep->create_date ? "" :
         ep->create_date == clause->create_date ? "SMX" : "DFX",

         !ep->derivation ? "" :
         ep->derivation == clause->derivation ? "SMX" : "DFX",
         clause->proof_size, clause->create_date
#endif
         );
    ClauseTSTPPrint(stdout, clause, true, true);
    printf("\n");
  }
  if (!ep->selected_literal_inited) {
    ep->selected_literal = sel; // initialize
    ep->selected_literal_inited = 1;
  } else if (ep->selected_literal != sel)
    printf("DIFFERENT SELECTED LITERAL: %d %ld %ld\n", epid, ep->selected_literal, sel);

  //ep->derivation = clause->derivation;
  //ep->create_date = clause->create_date;
  //ep->proof_size = clause->proof_size;
}


// could do this in a file, just assumed this is faster
#define CLAUSESTRBUF_SZ (100*1000*1000)
static char clauseStrBuf[CLAUSESTRBUF_SZ];
static size_t nclauseStrBuf;

#define MAX_NEW 100000
char already_seen2[UCLAUSE_SZ];
int drop_dups(int *aabuf, int naa) {
#if 1
  bzero(already_seen2, nclauseids);
  for (int i=0; i<naa; ++i) {
      if (aabuf[i]>=nclauseids) {
        printf("oops\n");
        exit(1);
      }
    already_seen2[aabuf[i]]=1;
  }
  if (dropdups==5)
    return naa;
#endif

  int foo[UCLAUSE_SZ];
  int *xbuf=dropdups<5? aabuf : foo;
  char already_seen[UCLAUSE_SZ];
  bzero(already_seen, nclauseids);
  int j=0;
  for (int i=0; i<naa; ++i) 
    if (!already_seen[aabuf[i]]) {
      if (aabuf[i]>=nclauseids) {
        printf("oops\n");
        exit(1);
      }
      already_seen[aabuf[i]]=1;
      xbuf[j]=aabuf[i];
      j++;
    }
  if (dropdups==6)
    return naa;
#if 1
  int new_naa=j;
  char already_seen3[UCLAUSE_SZ];
  bzero(already_seen3, nclauseids);
  for (int i=0; i<new_naa; ++i) {
    already_seen3[xbuf[i]]=1;
  }
  
  for (int i=0; i<nclauseids; ++i)
    if (already_seen[i]!=already_seen2[i]) {
        printf("oops2\n");
        exit(1);
      }
  for (int i=0; i<nclauseids; ++i)
    if (already_seen[i]!=already_seen3[i]) {
        printf("oops3\n"); 
        exit(1);
      }
  if (dropdups==7)
    return naa;
#endif
  
  if (new_naa<naa)
    printf("dropped %d dups\n", naa-new_naa);
  return new_naa;
}


static int register_now_now;
static void registerNewClauses(ProofControl_p control, ClauseSet_p *clausesets)
{
  printf("registerNewClauses\n"); fflush(stdout);

  static int stepn=0;

  int register_now=0;

  if (register_every_nsteps) {
    stepn++;
    if (stepn>=register_every_nsteps) {
      printf("register now!\n");
      register_now=1;
      stepn=0;
    }
  }
  if (register_now_now)
    register_now=1;
  
#define OBUFSZ (1000*1000)
  int aabuf[OBUFSZ];
  int pcbuf[OBUFSZ];
  MASKTP slbuf[OBUFSZ]; // used as bit masks 
  
  FILE *clauseBuf = fmemopen(clauseStrBuf, CLAUSESTRBUF_SZ-nclauseStrBuf-10000, "w");
  int naa=0;
  int npc=0;
  int newbuf[UCLAUSE_SZ];
  int nnew=0;

  int saw_demod = register_on_demodulator && processed_demodulator;
  for (int ii=0; ii<5; ii++) {
    int processed = (ii>0);
      
    ClauseSet_p clauseset = clausesets[ii];
    for(Clause_p handle = clauseset->anchor->succ; handle!=clauseset->anchor; handle = handle->succ) {
      int epid = handle->ident-LONG_MIN; 
      if (epid>=EPIDS_SZ) {
        printf("TOO MANY NOTES, HERR MOZART!  We only handle up to %d clauses\n",EPIDS_SZ);
        exit(1);
      }

      while (nepids <= epid) {
        struct EpInfo *ep = &epids[nepids];          
        ep->uclause = -1;
        ep->processed = -1; 
        nepids++;
      }

      struct EpInfo *ep = &epids[epid];

      if (save_clause_str_in_mem) {
          fseek(clauseBuf, nclauseStrBuf, SEEK_SET);        
          ClauseTSTPPrint(clauseBuf, handle, true, true);
          fflush(clauseBuf);
      }
          
      if (ep->uclause<0 ||
          register_now ||
          register_on_change ||
          (saw_demod && !strcmp(ep->clauseStr, &clauseStrBuf[nclauseStrBuf])) ||

          //not actually used or tested
          (register_on_processed && ep->processed != processed && printf("rop %d\n", epid))) {
        if (naa==OBUFSZ || npc==OBUFSZ) {
          printf("TOO MANY NEW!\n");
          exit(1);
        }
        ep->processed = processed; 
        if (save_clause_str_in_mem) {
          if (!(ep->uclause<0)) {
            if (print_selected_clause)
              printf("changed: %s\n", &clauseStrBuf[nclauseStrBuf]);
          }

          long pos = ftell(clauseBuf);
          //        if (nclauseStrBuf + 10*1000 > CLAUSESTRBUF_SZ) {
          if (pos + 10*1000 > CLAUSESTRBUF_SZ) {        
            printf("NO MORE BUF SPACE\n");
            exit(1);
          }
          clauseStrBuf[pos++] = 0;
          ep->clauseStr = &clauseStrBuf[nclauseStrBuf];
          //printf("new %d %ld\n", epid, pos-nclauseStrBuf); fflush(stdout);
          nclauseStrBuf = pos; // keep string
          fputs(ep->clauseStr, trailInputX);

          //fprintf(out, "%s",&clauseStrBuf[nclauseStrBuf]);
          //fprintf(out, "\n");
        } else
          ClauseTSTPPrint(trailInputX, handle, true, true);

        fprintf(trailInputX, "\n");
        
        if (get_literal_selection) {
          DoLiteralSelection(control, handle);
          MASKTP sel = clause_get_selected(handle);
          ep->selected_literal = sel;
          if (dbg_sellit)
            printf("sel: %d %ld\n", epid, sel);
          slbuf[nnew] = ep->selected_literal;
        }

        newbuf[nnew++] = epid;
      } else if (0&&strcmp(ep->clauseStr, &clauseStrBuf[nclauseStrBuf])) {
#if 0
        printf("clause %d changed (%d %d)\n", epid, ep->uclause, oldcnt[ep->uclause]);
        printf("%s\n", ep->clauseStr);
        printf("%s\n", &clauseStrBuf[nclauseStrBuf]);

        //newid[nc++] = epid;
        //fprintf(out, "%s",&clauseStrBuf[nclauseStrBuf]);
        //         fprintf(out, "\n");

        exit(1);
#if 0
        if (strlen(ep->clauseStr) >= strlen(&clauseStrBuf[nclauseStrBuf]))
          strcpy(ep->clauseStr, &clauseStrBuf[nclauseStrBuf]);
        else {
          ep->clauseStr = &clauseStrBuf[nclauseStrBuf];
          nclauseStrBuf = pos; // keep string
        }
#endif
#endif
      } else {
        if (processed)
          pcbuf[npc++] = ep->uclause;
        else {
          aabuf[naa++] = ep->uclause;
        }
      }
    }
  }
  fclose(clauseBuf);
  fprintf(trailInputX,"\n"); // tells Trail to stop reading and print the ids
  fflush(trailInputX);

  printf("wrote %d %d %d %ld\n", nnew, naa, npc, nclauseStrBuf);
  fflush(stdout);

  FILE *out = trailInputBinX;
  if (get_literal_selection) {
    MASKTP valp=nnew+1;
    for (int i=0; i<nnew; ++i) {
      MASKTP mask=slbuf[i];
      slbuf[i] = valp;
      if (dbg_sellit)
        printf("VALP: %i %ld %ld:", i, valp, mask);
      for (int j=0; j<sizeof(long); ++j) 
        if (mask&(1<<j)) {
          if (dbg_sellit)
            printf(" %ld:%d", valp, j);
          slbuf[valp++] = j;
          if (valp>=OBUFSZ) {
            printf("TOO MANY SEL VALS\n");
            exit(1);
          }
        }
      if (dbg_sellit)
        printf("\n");
    }
    slbuf[nnew] = valp;
    fwrite(&valp, sizeof(MASKTP), 1, out);
    fwrite(slbuf, sizeof(MASKTP), valp, out);

    fflush(out);
    printf("wrote selected_lit\n"); fflush(stdout);
  }

  // doing all together to avoid lots of context switches (? does it really matter?)
  for (int i=0; i<nnew; ++i) {
    int epid=newbuf[i];
    struct EpInfo *ep = &epids[epid];
    
    int clauseid;
    if (!fscanf(trailOutputX, "%d", &clauseid)) {
      printf("Error reading clauseid!\n");
      exit(1);      
    }
    if (clauseid >= UCLAUSE_SZ) {
      printf("clauseid too large\n");
      exit(1);
    }
    if (nclauseids<=clauseid)
      nclauseids=clauseid+1;
    //else
    //printf("seen before: %d\n", clauseid);
    
    //printf("assigning %d %d\n", epid, clauseid);
    //    fflush(stdout);
    ep->uclause=clauseid;
    if (ep->processed)
      pcbuf[npc++] = clauseid;
    else {
      slbuf[naa] = ep->selected_literal;
      aabuf[naa++] = clauseid;
    }
  }

  static int dx=0;
  int dd=dropdups;

  // 1: drop all
  switch (dd) {
  case 0:  break; // do nothing
  case 1:  break; // dropdup all
  case 2:  if (dx>0)  dd=0; break; // only dropdup first
  case 3:  if (dx==0) dd=0; break; // only dropdup after first
  case 4:  dd=0; break; // python code dropdup
  case 5:  break;       // check 1, do nothing
  case 6:  break;       // check 2, do nothing
  case 7:  break;       // check 3, do nothing
    
  default:
    printf("bad dd: %d\n", dd);
    exit(1);
  }
  dx=1;
    
  if (dd) 
    naa = drop_dups(aabuf, naa);
  fwrite(&naa, sizeof(int), 1, out);
  fwrite(aabuf, sizeof(int), naa, out); 
  fflush(out);
  printf("wrote newunprocessed %d\n", naa); fflush(stdout);

  if (dd) 
    npc = drop_dups(pcbuf, npc);
  fwrite(&npc, sizeof(int), 1, out);
  fwrite(pcbuf, sizeof(int), npc, out); 
  fflush(out);
  printf("wrote newprocessed %d\n", npc); fflush(stdout);

  if (dd>1) 
    printf("dropdups foo %d\n", already_seen2[0]);
  
  processed_demodulator=0;
}

#include <time.h>       // for clock_t, clock(), CLOCKS_PER_SEC
static clock_t begin_time;

static Clause_p maybe_select_single_lit(ProofState_p state) {
  static int ntimes=0;

  if (ntimes>=100)
    return 0;
  
  ClauseSet_p clauseset = state->unprocessed;
  for(Clause_p handle = clauseset->anchor->succ; handle!=clauseset->anchor; handle = handle->succ) {
    //if (handle->literals && !handle->literals->next) {
    if (ClauseIsUnit(handle)) {
      //int sel = EqnIsOriented((handle)->literals);
      int sel = ClauseIsNegative(handle) || !ClauseIsEquational(handle);
      printf("auto-selecting single-lit clause: %d (%d %d %d %d %d %d) ",
             sel,
             ClauseIsDemodulator(handle),             
             ClauseIsUnit(handle),
             ClauseIsGround(handle),
             ClauseIsNegative(handle),
             ClauseIsEquational(handle),
             EqnIsOriented((handle)->literals));
      ClauseTSTPCorePrint(stdout, handle, true);
      printf("\n");
      fflush(stdout);
      if (sel) {
        ntimes++;
        return handle;
      }
    }
  }
  return 0;
}

static void writeTrailInput(ProofState_p state, ProofControl_p control) {
  
  if (!begin_time)
    begin_time = clock();
  printf("writeTrailInput %f\n", (double)(clock() - begin_time) / CLOCKS_PER_SEC); 

#if 0
  for (int epid=0; epid<nepids; ++epid) {
    struct EpInfo *ep = &epids[epid];
    int uclause=ep->uclause;
    if (uclause>=0 &&
        countBufs[uclause]
        //!countBufs[0][whichbuf][uclause]
        //!countBufs[1][whichbuf][uclause]
        ) {) {
      printf("deleting %d %d\n", epid, uclause);
      ep->uclause=-1;
    }
  }
#endif
  
#if 0
    for (int pr=0; pr<=1; ++pr) {
    char *cnt = countBufs[pr][whichbuf];  
    bzero(cnt, nclauseids*sizeof(int));
  }
#else
    bzero(countBufs, nclauseids*sizeof(int));
#endif

    
  printf("register\n");  fflush(stdout);
  ClauseSet_p  all_sets[] = {state->unprocessed,
    state->processed_pos_eqns, state->processed_pos_rules,
    state->processed_neg_units, state->processed_non_units};
  registerNewClauses(control, all_sets);

  //whichbuf = !whichbuf;
}

#if 0

static int writeClauseIds(int processed, int whichbuf)
{
  printf("writeClauseIds %d %d\n", processed, whichbuf);
  int nclauses=0;
  FILE *out=trailInputBinX;

  //char *cnt = countBufs[processed][whichbuf];
  char *cnt = countBufs;
  int buf[UCLAUSE_SZ];
  
  for (int uclause=0; uclause<nclauseids; uclause++) {
    /* printf("diff%d %d %d %d\n", add, uclause, oldcnt[uclause], cnt[uclause]); */
    if (cnt[uclause]) {
      //printf("clid %d\n", uclause);
      buf[nclauses++] = uclause;
      //fprintf(out, " %d", uclause); nclauses++;
    }
  }
  fwrite(&nclauses, sizeof(int), 1, out);
  fwrite(buf, sizeof(int), nclauses, out);
  fflush(out);
  
  return nclauses;
}

static void registerNewClauses2(int processed, int whichbuf, int *newid, int nc)
{
  printf("registerNewClauses2 %d %d\n", processed, whichbuf);
  fflush(stdout);   

  
  char *cnt = countBufs[processed][whichbuf];
  FILE *in = trailOutputX;
  for (int i=0; i<nc; i++) {
    int epid = newid[i];
    int clauseid;
    if (!fscanf(in, "%d", &clauseid)) {
      printf("Error reading clauseid!\n");
      exit(1);      
    }
    /*    if (epid<0) {
      epid = -epid;
      epids[epid].old_uclause = epids[epid].uclause;
      }*/
    epids[epid].uclause=clauseid;
    if (clauseid >= UCLAUSE_SZ) {
      printf("clauseid too large\n");
      exit(1);
    }
    if (nclauseids<=clauseid)
      nclauseids=clauseid+1;
    
    cnt[clauseid]++; 
  }
  // no need for signal
}


static int writeAllClauses(ClauseSet_p clauseset, int processed, int whichbuf) 
{
  printf("writeAllClauses %d %d\n", processed, whichbuf);
  int nclauses=0;
  FILE *out=trailInputX;

  char *cnt = countBufs[processed][whichbuf];
  
  for(Clause_p handle = clauseset->anchor->succ; handle!=clauseset->anchor; handle = handle->succ) {
    int epid = handle->ident-LONG_MIN;
    struct EpInfo *ep = &epids[epid];
    int uclause = ep->uclause;
    if (cnt[uclause]) {
      fprintf(out, " %d", uclause);         
      nclauses++;
    }
  }
  printf("wac %d\n",nclauses);
  return nclauses;
}

 #if 0
static int writeAllClauseDiffs(ClauseSet_p clauseset, int processed, int whichbuf, int *buf) 
{
  printf("writeAllClauseDiffs %d %d\n", processed, whichbuf);
  int nclauses=0;
  FILE *out=trailInputX;

  char *cnt = countBufs[processed][whichbuf];
  char *oldcnt = countBufs[processed][!whichbuf];
  
  for(Clause_p handle = clauseset->anchor->succ; handle!=clauseset->anchor; handle = handle->succ) {
    int epid = handle->ident-LONG_MIN;
    struct EpInfo *ep = &epids[epid];
    int uclause = ep->uclause;
    if (!oldcnt[uclause] && cnt[uclause]) {
      ClauseTSTPPrint(out, handle, true, true); fprintf(out, "\n");
      buf[nclauses++] = ep->trid;
      nclauses++;
    }
  }
  printf("wac %d\n",nclauses);
  return nclauses;
}
#endif

static int writeClauseDiffs(int add, int processed, int whichbuf)
{
  printf("writeClauseDiffs%d %d %d\n", add, processed, whichbuf);
  int nclauses=0;
  FILE *out=trailInputX;

  char *cnt = countBufs[processed][whichbuf];
  char *oldcnt = countBufs[processed][!whichbuf];

  for (int uclause=0; uclause<nclauseids; uclause++) {
    /* printf("diff%d %d %d %d\n", add, uclause, oldcnt[uclause], cnt[uclause]); */
    if (add
        ? !oldcnt[uclause] && cnt[uclause]
        : oldcnt[uclause] && !cnt[uclause]) {
      printf("%s %d\n", add?"add":"del", uclause);
      //buf[nclauses++] = uclause;
      fprintf(out, " %d", uclause); nclauses++;
    }
  }

  printf("wcd %d\n",nclauses);
  return nclauses;
}

static void writeTrailInputX(ProofState_p state) {
  FILE *out=trailInputX;
#define OBUFSZ (1000*1000)
  //  char buf[OBUFSZ];
  //  int ibuf[OBUFSZ];
  static int whichbuf = 0;

  for (int pr=0; pr<=1; ++pr) {
    char *cnt = countBufs[pr][whichbuf];  
    bzero(cnt, nclauseids);
  }
    
  printf("writeTrailInput %d\n",whichbuf);  fflush(out);
  // I'm pretty sure that you MUST flush one pipe before you start to write to another,
  // otherwise Trail will hang.

  ClauseSet_p  processed_clauses[] = {state->processed_pos_eqns, state->processed_pos_rules,
                                      state->processed_neg_units, state->processed_non_units};

  fprintf(out, "0\n"); // not stopped
  fflush(out); // important
  
  //int nnew=0;
  int newid[MAX_NEW];
  
  int nc0 = registerNewClauses1(state->unprocessed, 0, whichbuf, newid);
  int nc1=0;
  for (int i=0; i<4; ++i) {
    //  the same epid can appear in more than one of these lists
    nc1 += registerNewClauses1(processed_clauses[i], 1, whichbuf, newid+nc0+nc1);
  }
  fprintf(out, "\n"); // signal end
  fflush(out); // important
  
  registerNewClauses2(0, whichbuf, newid, nc0);
  registerNewClauses2(1, whichbuf, newid+nc0, nc1);
  fflush(stdout);

#if 0
  int d10 = writeClauseDiffs(1, 0, whichbuf);
  fprintf(out, "\n");
  int d00 = writeClauseDiffs(0, 0, whichbuf);
  fprintf(out, "\n");

  int d11 = writeClauseDiffs(1, 1, whichbuf);
  fprintf(out, "\n");
  int d01 = writeClauseDiffs(0, 1, whichbuf);
  fprintf(out, "\n");
  fflush(stdout);
  fflush(out); // important

  if (! (d10||d00||d11||d01)) {
    printf("NO DIFFS!\n");
    int somediff=0;
    
    for (int pr=0; pr<=1; ++pr) {
      for (int uclause=0; uclause<=maxclauseid; uclause++) {
        int cnt0 = countBufs[pr][0][uclause];
        int cnt1 = countBufs[pr][1][uclause];
  
        if (cnt0 != cnt1) {
          printf("cd %d %d %d %d\n", pr, uclause, cnt0, cnt1);
          somediff=1;
        }
      }
    }
    if (!somediff)
      exit(1);
  }
#else
  writeClauseIds(0, whichbuf);
  writeClauseIds(1, whichbuf);
#endif
  
  static int dbgchk= -1;
  if (dbgchk == -1) {
    dbgchk = getenv("TRAIL_CHECK_IDS")!=0;
  }
  if (dbgchk) {
    writeAllClauses(state->unprocessed, 0, whichbuf);
    fprintf(out, "\n");
    for (int i=0; i<4; ++i) 
      writeAllClauses(processed_clauses[i], 1, whichbuf);
    fprintf(out, "\n");
    fflush(stdout);
    fflush(out); // important
  }
  
  fflush(stdout);

  whichbuf = !whichbuf;
}
#endif

int trail_stopping=0;
// you MUST write these IN THE SAME ORDER as writeTrailInput
// ... unless we were told to stop, in which case no more input is expected.
// by not exiting immediately, we get the E summary.
void writeDummyTrailInput() {
  FILE *out;
  // NUTS - repeated from above
  char *trailOutputName = "Trail2E.pipe";
  int trailOutputExists = (access(trailOutputName, F_OK ) == 0);

  if (trailOutputExists) {
    if (trail_stopping) {
      printf("Not writing Trail dummy output!\n"); fflush(stdout);
    } else {
out = trailInputBinX;
        int nup=0;
        printf("writing dummy\n"); fflush(stdout);

                out = trailInputX;
        fprintf(out,"\n");
              fflush(out);

      out = trailInputBinX;
              fwrite(&nup, sizeof(int), 1, out);

  if (get_literal_selection) {
    MASKTP n=0;
                  fwrite(&n, sizeof(MASKTP), 1, out);
  }
        fflush(out);

      
                out = trailInputX;
        fprintf(out,"\n");
              fflush(out);
              
      out = trailInputBinX;
      fwrite(&nup, sizeof(int), 1, out);
      fflush(out);
              
      /*
        out = trailInputBinX;
        int nup=0;
        printf("writing dummy\n"); fflush(stdout);
        fwrite(&nup, sizeof(int), 1, out);
        //fclose(out);
        fflush(out);
      */
      /* out = fopen("newunprocessed", "w");   fclose(out); */

      //out = trailInputX;
      //fprintf(out, "1\n"); // stopped
#if 0
                fprintf(out,"\n"); // end new clauses

                fprintf(out,"\n");
                fprintf(out,"\n");
                fprintf(out,"\n");
                fprintf(out,"\n");
              fflush(out);
              
              /*      out = trailInputBinX;
      fwrite(&nup, sizeof(int), 1, out);
      fflush(out);
              */
                out = trailInputX;
        fprintf(out,"\n");
              fflush(out);
#endif

      printf("writing dummy2\n");
    }
  }
}


static Clause_p readTrailOutput(ProofState_p state, FILE *trailInput, FILE *trailOutput) {
   Clause_p trailClause = NULL;
   
     if (trailOutput!=NULL) {
    	  /* printf("reading from trailOutput\n"); fflush(stdout); */

       static pid_t mypid = 0;
       if (!mypid)
         mypid = getpid();

          int ident=FORK_COMMAND;
          while (ident == FORK_COMMAND) {
            char cbuf[1024];
            while (1) {
              if (fgets(cbuf, 1024, trailOutput)==NULL) {
                printf("NO output from Trail!\n");
                fflush(stdout);                
                exit(1);
                sleep(1);
              } else if (cbuf[0]!='\n') {
                break;
              }
            }
            if (sscanf(cbuf, "%d", &ident)!=1) {
              printf("Reading line from trail failed! %d >%s<\n", mypid, cbuf);
              fflush(stdout);              
              exit(1);
            }

            printf("eprover read ident %i %f\n",
                   ident, (double)(clock() - begin_time) / CLOCKS_PER_SEC);
            fflush(stdout);

            if (ident == STOP_COMMAND) {
              printf("stopping %d!\n", mypid);
              fflush(stdout);
              trail_stopping=1; // nuts
              return NULL;
            }

            if (ident == FORK_COMMAND) {
              // read this before forking, to ensure that child and parent don't both try to read
              if (fgets(cbuf, 1024, trailOutput)==NULL) {
                printf("Reading dir from trail failed!\n");
                exit(1);
              }
              int dlen = strlen(cbuf);
              if (cbuf[dlen-1]!='\n') {
                printf("Reading dir from trail looks incomplete!! >%s<\n", cbuf);
                printf("Must be the only thing on the line\n");
                exit(1);
              }
              cbuf[dlen-1]=0;
              printf("forking to dir: >%s<\n", cbuf); fflush(stdout);
              
              pid_t child_pid=fork();
              
              if (child_pid == 0) {
                mypid = getpid();
                if (strcmp(cbuf,".")==0) {
                  printf("Child forked %d in place\n", mypid); fflush(stdout);
                  fprintf(trailInput, "%d\n", mypid); fflush(trailInput);
                  printf("child sent its own pid\n"); fflush(stdout);
                } else {
                printf("Child forked %d to %s\n", mypid,cbuf); fflush(stdout);
                if (chdir(cbuf)) { // unistd
                  printf("dir doesn't exist! child exiting. >%s<\n", cbuf);
                  exit(1);
                }

                if (fopen("eprover.txt", "r")) {
                  printf("eprover.txt already exists! child eprover fork failed.\n");
                  exit(1);
                }
                if (freopen("eprover.txt","w",stdout)==NULL) { // best if this matches the name from launch-ee.sh
                  printf("Can't reopen stdout to eprover.txt!\n");
                  exit(1);
                }
                if (freopen("eprover.txt","w",stderr)==NULL) { // best if this matches the name from launch-ee.sh
                  printf("Can't reopen stderr to eprover.txt!\n");
                  exit(1);
                }
                reopenTrailPipes(trailInput, trailOutput);

                //fprintf(trailInput, "%d\n", mypid); fflush(trailInput); // send pid just as a test
                /*
                int errn = flock(1, LOCK_EX|LOCK_NB); // sys/file.h
                if (errn) {
                  printf("Can't lock eprover.txt!  Some other process must be running in this dir. %d\n", errn);
                  exit(1);
                  }*/
                printf("child reading commands\n"); fflush(stdout);
                // we never explicitly release this lock.
                // there is no provision for re-using dirs.
                }
              } else if (child_pid < 0) {
                printf("fork failed! %d %d\n", mypid, child_pid);
                exit(1);
              } else {
                printf("parent forked %d -> %d\n", mypid, child_pid);
                // parent waits
                fflush(stdout);
                if (strcmp(cbuf,".")==0) {
                  int status = 0;
                  printf("parent %d waiting for %d\n", mypid, child_pid); fflush(stdout);
                  pid_t wpid = wait(&status);
                  if (wpid != child_pid) {
                    printf("bad wpid! %d %d %d\n", wpid, child_pid, mypid);
                    if (wpid == -1)
                      printf("errno(%d): %s\n", errno, strerror(errno));
                    exit(1);
                  }
                } else {
                  fprintf(trailInput, "%d\n", child_pid); fflush(trailInput);
                  printf("parent sent child pid\n"); fflush(stdout);
                }                  
                printf("parent %d continuing from %d\n", mypid, child_pid);fflush(stdout);
                // parent loops
              }
            }
          }

          trailClause = getclause(state->unprocessed, ident);
          if (print_selected_clause) {
            printf("selact: ");
            ClauseTSTPPrint(stdout, trailClause, true, true);
            printf("\n");
          }
          if (check_selected_clause) {
            ClauseTSTPCorePrint(selectedClauseFile, trailClause, true);
            fprintf(selectedClauseFile,"\n");
          }
          fflush(trailInput);
          printf("end readInput\n");
          register_now_now=0;
          if (re_register_on_single_lit_action && trailClause->literals && !trailClause->literals->next) {
            // single-literal clauses rewrite actions
            // I thought it was correct to check for demodulators, but that didn't work
            printf("re-registering:  ");
            ClauseTSTPCorePrint(stdout, trailClause, true);
            printf("\n");
            register_now_now=1;
          }
    	  fflush(stdout);
      }
     return trailClause;
}

#if 0
static double might_match(Term_p term1, Term_p term2)
{
   int i;

   if (TermIsVar(term1) || TermIsVar(term2)) {
     return 1;
   }

   if ((term1->f_code != term2->f_code) || (term1->arity != term2->arity)) {
     return 0;
   }

   for (i=0; i<term1->arity; i++) 
   {
     if (!might_match(term1->args[i],term2->args[i]))
       return 0;
   }
   return 1;
}

static int might_match_ClauseSet(ProofState_p state, Clause_p clause)
{
  int nmatches=0;
  ClauseSet_p  processed_clauses[] = {state->processed_pos_eqns, state->processed_pos_rules,
                                      state->processed_neg_units, state->processed_non_units};
  for (int i=0; i<4; ++i) {
    ClauseSet_p clauseset = processed_clauses[i];

    for(Clause_p handle = clauseset->anchor->succ; handle!=clauseset->anchor; handle = handle->succ) {
      Eqn_p eqn1, eqn2;
    
      for(eqn1 = clause->literals; eqn1; eqn1 = eqn1->next)
        for(eqn2 = handle->literals; eqn2; eqn2 = eqn2->next)
          if (!EqnIsEquLit(eqn1) &&
              !EqnIsEquLit(eqn2) &&
              EqnIsPositive(eqn1) != EqnIsPositive(eqn2) &&
              might_match(eqn1->lterm, eqn2->lterm)) {
            /*
            TBPrintTerm(stdout, eqn1->bank, eqn1->lterm, 1);
            printf("\n");
            TBPrintTerm(stdout, eqn2->bank, eqn2->lterm, 1);
            printf("\n");
            */
            nmatches++;
          }
    }
  }
  return nmatches;
}

   } else if (0) {
#if 1
     clause = NULL;
     int bmc=0;
     ClauseSet_p clauseset=state->unprocessed; 
     for(Clause_p handle = clauseset->anchor->succ; handle!=clauseset->anchor; handle = handle->succ) {
       int mc=might_match_ClauseSet(state, handle);
       if (!clause || mc>bmc) {
         bmc=mc;
         clause = handle;
       }         
#if 0
       printf("%d", mc);
       if (handle==clause)
         printf("**\n");
       else
         printf("\n");
#endif
     }
#endif

#endif
