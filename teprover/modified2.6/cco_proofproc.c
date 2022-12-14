#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <sys/file.h>
/*-----------------------------------------------------------------------

  File  : cco_proofproc.c

  Author: Stephan Schulz

  Contents

  Functions realizing the proof procedure.

  Copyright 1998--2018 by the author.
  This code is released under the GNU General Public Licence and
  the GNU Lesser General Public License.
  See the file COPYING in the main E directory for details..
  Run "eprover -h" for contact information.

Changes

<1> Mon Jun  8 11:47:44 MET DST 1998
    New

-----------------------------------------------------------------------*/

#include "cco_proofproc.h"
#include <picosat.h>
#include <cco_ho_inferences.h>



/*---------------------------------------------------------------------*/
/*                        Global Variables                             */
/*---------------------------------------------------------------------*/

PERF_CTR_DEFINE(ParamodTimer);
PERF_CTR_DEFINE(BWRWTimer);


/*---------------------------------------------------------------------*/
/*                      Forward Declarations                           */
/*---------------------------------------------------------------------*/


/*---------------------------------------------------------------------*/
/*                         Internal Functions                          */
/*---------------------------------------------------------------------*/

/*-----------------------------------------------------------------------
//
// Function: document_processing()
//
//   Document processing of the new given clause (depending on the
//   output level).
//
// Global Variables: OutputLevel, GlobalOut (read only)
//
// Side Effects    : Output
//
/----------------------------------------------------------------------*/

void document_processing(Clause_p clause)
{
   if(OutputLevel)
   {
      if(OutputLevel == 1)
      {
         putc('\n', GlobalOut);
         putc('#', GlobalOut);
         ClausePrint(GlobalOut, clause, true);
         putc('\n', GlobalOut);
      }
      DocClauseQuoteDefault(6, clause, "new_given");
   }
}

/*-----------------------------------------------------------------------
//
// Function: check_ac_status()
//
//   Check if the AC theory has been extended by the currently
//   processed clause, and act accordingly.
//
// Global Variables: -
//
// Side Effects    : Changes AC status in signature
//
/----------------------------------------------------------------------*/

static void check_ac_status(ProofState_p state, ProofControl_p
                            control, Clause_p clause)
{
   if(control->heuristic_parms.ac_handling!=NoACHandling)
   {
      bool res;
      res = ClauseScanAC(state->signature, clause);
      if(res && !control->ac_handling_active)
      {
         control->ac_handling_active = true;
         if(OutputLevel)
         {
            SigPrintACStatus(GlobalOut, state->signature);
            fprintf(GlobalOut, "# AC handling enabled dynamically\n");
         }
      }
   }
}


/*-----------------------------------------------------------------------
//
// Function: remove_subsumed()
//
//   Remove all clauses subsumed by subsumer from set, kill their
//   children. Return number of removed clauses.
//
// Global Variables: -
//
// Side Effects    : Changes set, memory operations.
//
/----------------------------------------------------------------------*/

static long remove_subsumed(GlobalIndices_p indices,
                            FVPackedClause_p subsumer,
                            ClauseSet_p set,
                            ClauseSet_p archive)
{
   Clause_p handle;
   long     res;
   PStack_p stack = PStackAlloc();

   res = ClauseSetFindFVSubsumedClauses(set, subsumer, stack);

   while(!PStackEmpty(stack))
   {
      handle = PStackPopP(stack);
      //printf("# XXX Removing (remove_subumed()) %p from %p = %p\n", handle, set, handle->set);
      if(ClauseQueryProp(handle, CPWatchOnly))
      {
         DocClauseQuote(GlobalOut, OutputLevel, 6, handle,
                        "extract_wl_subsumed", subsumer->clause);

      }
      else
      {
         DocClauseQuote(GlobalOut, OutputLevel, 6, handle,
                        "subsumed", subsumer->clause);
      }
      GlobalIndicesDeleteClause(indices, handle);
      ClauseSetExtractEntry(handle);
      ClauseSetProp(handle, CPIsDead);
      ClauseSetInsert(archive, handle);
   }
   PStackFree(stack);
   return res;
}


/*-----------------------------------------------------------------------
//
// Function: eliminate_backward_rewritten_clauses()
//
//   Remove all processed clauses rewritable with clause and put
//   them into state->tmp_store.
//
// Global Variables: -
//
// Side Effects    : Changes clause sets
//
/----------------------------------------------------------------------*/

static bool
eliminate_backward_rewritten_clauses(ProofState_p
                                     state, ProofControl_p control,
                                     Clause_p clause, SysDate *date)
{
   long old_lit_count = state->tmp_store->literals,
      old_clause_count= state->tmp_store->members;
   bool min_rw = false;

   PERF_CTR_ENTRY(BWRWTimer);
   if(ClauseIsDemodulator(clause))
   {
      SysDateInc(date);
      if(state->gindices.bw_rw_index)
      {
         min_rw = RemoveRewritableClausesIndexed(control->ocb,
                                                 state->tmp_store,
                                                 state->archive,
                                                 clause, *date, &(state->gindices));

      }
      else
      {
         min_rw = RemoveRewritableClauses(control->ocb,
                                          state->processed_pos_rules,
                                          state->tmp_store,
                                          state->archive,
                                          clause, *date, &(state->gindices))
            ||min_rw;
         min_rw = RemoveRewritableClauses(control->ocb,
                                          state->processed_pos_eqns,
                                          state->tmp_store,
                                          state->archive,
                                          clause, *date, &(state->gindices))
            ||min_rw;
         min_rw = RemoveRewritableClauses(control->ocb,
                                          state->processed_neg_units,
                                          state->tmp_store,
                                          state->archive,
                                          clause, *date, &(state->gindices))
            ||min_rw;
         min_rw = RemoveRewritableClauses(control->ocb,
                                          state->processed_non_units,
                                          state->tmp_store,
                                          state->archive,
                                          clause, *date, &(state->gindices))
            ||min_rw;
      }
      state->backward_rewritten_lit_count+=
         (state->tmp_store->literals-old_lit_count);
      state->backward_rewritten_count+=
         (state->tmp_store->members-old_clause_count);

      if(control->heuristic_parms.detsort_bw_rw)
      {
         ClauseSetSort(state->tmp_store, ClauseCmpByStructWeight);
      }
   }
   PERF_CTR_EXIT(BWRWTimer);
   /*printf("# Removed %ld clauses\n",
     (state->tmp_store->members-old_clause_count)); */
   return min_rw;
}


/*-----------------------------------------------------------------------
//
// Function: eliminate_backward_subsumed_clauses()
//
//   Eliminate subsumed processed clauses, return number of clauses
//   deleted.
//
// Global Variables: -
//
// Side Effects    : Changes clause sets.
//
/----------------------------------------------------------------------*/

static long eliminate_backward_subsumed_clauses(ProofState_p state,
                                                FVPackedClause_p pclause)
{
   long res = 0;

   if(ClauseLiteralNumber(pclause->clause) == 1)
   {
      if(pclause->clause->pos_lit_no)
      {
         /* A unit rewrite rule that is a variant of an old rule is
            already subsumed by the older one.
            A unit rewrite rule can never subsume an unorientable
            equation (else it would be unorientable itself). */
         if(!ClauseIsRWRule(pclause->clause))
         {
            res += remove_subsumed(&(state->gindices), pclause,
                                   state->processed_pos_rules,
                                   state->archive);
            res += remove_subsumed(&(state->gindices), pclause,
                                   state->processed_pos_eqns,
                                   state->archive);
         }
         res += remove_subsumed(&(state->gindices), pclause,
                                state->processed_non_units,
                                state->archive);
      }
      else
      {
         res += remove_subsumed(&(state->gindices), pclause,
                                state->processed_neg_units,
                                state->archive);
         res += remove_subsumed(&(state->gindices), pclause,
                                state->processed_non_units,
                                state->archive);
      }
   }
   else
   {
      res += remove_subsumed(&(state->gindices), pclause,
                             state->processed_non_units,
                             state->archive);
   }
   state->backward_subsumed_count+=res;
   return res;
}



/*-----------------------------------------------------------------------
//
// Function: eliminate_unit_simplified_clauses()
//
//   Perform unit-back-simplification on the proof state.
//
// Global Variables: -
//
// Side Effects    : Potentially changes and moves clauses.
//
/----------------------------------------------------------------------*/

static void eliminate_unit_simplified_clauses(ProofState_p state,
                                              Clause_p clause)
{
   if(ClauseIsRWRule(clause)||!ClauseIsUnit(clause))
   {
      return;
   }
   ClauseSetUnitSimplify(state->processed_non_units, clause,
                         state->tmp_store,
                         state->archive,
                         &(state->gindices));
   if(ClauseIsPositive(clause))
   {
      ClauseSetUnitSimplify(state->processed_neg_units, clause,
                            state->tmp_store,
                            state->archive,
                            &(state->gindices));
   }
   else
   {
      ClauseSetUnitSimplify(state->processed_pos_rules, clause,
                            state->tmp_store,
                            state->archive,
                            &(state->gindices));
      ClauseSetUnitSimplify(state->processed_pos_eqns, clause,
                            state->tmp_store,
                            state->archive,
                            &(state->gindices));
   }
}

/*-----------------------------------------------------------------------
//
// Function: eliminate_context_sr_clauses()
//
//   If required by control, remove all
//   backward-contextual-simplify-reflectable clauses.
//
// Global Variables: -
//
// Side Effects    : Moves clauses from state->processed_non_units
//                   to state->tmp_store
//
/----------------------------------------------------------------------*/

static long eliminate_context_sr_clauses(ProofState_p state,
                                         ProofControl_p control,
                                         Clause_p clause)
{
   if(!control->heuristic_parms.backward_context_sr)
   {
      return 0;
   }
   return RemoveContextualSRClauses(state->processed_non_units,
                                    state->tmp_store,
                                    state->archive,
                                    clause,
                                    &(state->gindices));
}

/*-----------------------------------------------------------------------
//
// Function: check_watchlist()
//
//   Check if a clause subsumes one or more watchlist clauses, if yes,
//   set appropriate property in clause and remove subsumed clauses.
//
// Global Variables: -
//
// Side Effects    : As decribed.
//
/----------------------------------------------------------------------*/


void check_watchlist(GlobalIndices_p indices, ClauseSet_p watchlist,
                     Clause_p clause, ClauseSet_p archive,
                     bool static_watchlist)
{
   FVPackedClause_p pclause;
   long removed;

   if(watchlist)
   {
      pclause = FVIndexPackClause(clause, watchlist->fvindex);
      // printf("# check_watchlist(%p)...\n", indices);
      ClauseSubsumeOrderSortLits(clause);
      // assert(ClauseIsSubsumeOrdered(clause));

      clause->weight = ClauseStandardWeight(clause);

      if(static_watchlist)
      {
         Clause_p subsumed;

         subsumed = ClauseSetFindFirstSubsumedClause(watchlist, clause);
         if(subsumed)
         {
            ClauseSetProp(clause, CPSubsumesWatch);
         }
      }
      else
      {
         if((removed = remove_subsumed(indices, pclause, watchlist, archive)))
         {
            ClauseSetProp(clause, CPSubsumesWatch);
            if(OutputLevel == 1)
            {
               fprintf(GlobalOut,"# Watchlist reduced by %ld clause%s\n",
                       removed,removed==1?"":"s");
            }
            // ClausePrint(GlobalOut, clause, true); printf("\n");
            DocClauseQuote(GlobalOut, OutputLevel, 6, clause,
                           "extract_subsumed_watched", NULL);   }
      }
      FVUnpackClause(pclause);
      // printf("# ...check_watchlist()\n");
   }
}


/*-----------------------------------------------------------------------
//
// Function: simplify_watchlist()
//
//   Simplify all clauses in state->watchlist with processed positive
//   units from state. Assumes that all those clauses are in normal
//   form with respect to all clauses but clause!
//
// Global Variables: -
//
// Side Effects    : Changes watchlist, introduces new rewrite links
//                   into the term bank!
//
/----------------------------------------------------------------------*/

void simplify_watchlist(ProofState_p state, ProofControl_p control,
                        Clause_p clause)
{
   ClauseSet_p tmp_set;
   Clause_p handle;
   long     removed_lits;

   if(!ClauseIsDemodulator(clause))
   {
      return;
   }
   // printf("# simplify_watchlist()...\n");
   tmp_set = ClauseSetAlloc();

   if(state->wlindices.bw_rw_index)
   {
      // printf("# Simpclause: "); ClausePrint(stdout, clause, true); printf("\n");
      RemoveRewritableClausesIndexed(control->ocb,
                                     tmp_set, state->archive,
                                     clause, clause->date,
                                     &(state->wlindices));
      // printf("# Simpclause done\n");
   }
   else
   {
      RemoveRewritableClauses(control->ocb, state->watchlist,
                              tmp_set, state->archive,
                              clause, clause->date,
                              &(state->wlindices));
   }
   while((handle = ClauseSetExtractFirst(tmp_set)))
   {
      // printf("# WL simplify: "); ClausePrint(stdout, handle, true);
      // printf("\n");
      ClauseComputeLINormalform(control->ocb,
                                state->terms,
                                handle,
                                state->demods,
                                control->heuristic_parms.forward_demod,
                                control->heuristic_parms.prefer_general);
      removed_lits = ClauseRemoveSuperfluousLiterals(handle);
      if(removed_lits)
      {
         DocClauseModificationDefault(handle, inf_minimize, NULL);
      }
      if(control->ac_handling_active)
      {
         ClauseRemoveACResolved(handle);
      }
      handle->weight = ClauseStandardWeight(handle);
      ClauseMarkMaximalTerms(control->ocb, handle);
      ClauseSetIndexedInsertClause(state->watchlist, handle);
      // printf("# WL Inserting: "); ClausePrint(stdout, handle, true); printf("\n");
      GlobalIndicesInsertClause(&(state->wlindices), handle);
   }
   ClauseSetFree(tmp_set);
   // printf("# ...simplify_watchlist()\n");
}


/*-----------------------------------------------------------------------
//
// Function: generate_new_clauses()
//
//   Apply the generating inferences to the proof state, putting new
//   clauses into state->tmp_store.
//
// Global Variables: -
//
// Side Effects    : Changes proof state as described.
//
/----------------------------------------------------------------------*/

static void generate_new_clauses(ProofState_p state, ProofControl_p
                                 control, Clause_p clause, Clause_p tmp_copy)
{
   ComputeHOInferences(state,control,tmp_copy,clause);
   if(control->heuristic_parms.enable_eq_factoring)
   {
      state->factor_count+=
         ComputeAllEqualityFactors(state->terms, control->ocb, clause,
                                   state->tmp_store, state->freshvars);
   }
   state->resolv_count+=
      ComputeAllEqnResolvents(state->terms, clause, state->tmp_store,
                              state->freshvars);

   if(control->heuristic_parms.enable_neg_unit_paramod
      ||!ClauseIsUnit(clause)
      ||!ClauseIsNegative(clause))
   { /* Sometime we want to disable paramodulation for negative units */
      PERF_CTR_ENTRY(ParamodTimer);
      if(state->gindices.pm_into_index)
      {
         state->paramod_count+=
            ComputeAllParamodulantsIndexed(state->terms,
                                           control->ocb,
                                           state->freshvars,
                                           tmp_copy,
                                           clause,
                                           state->gindices.pm_into_index,
                                           state->gindices.pm_negp_index,
                                           state->gindices.pm_from_index,
                                           state->tmp_store,
                                           control->heuristic_parms.pm_type);
      }
      else
      {
         state->paramod_count+=
            ComputeAllParamodulants(state->terms, control->ocb,
                                    tmp_copy, clause,
                                    state->processed_pos_rules,
                                    state->tmp_store, state->freshvars,
                                    control->heuristic_parms.pm_type);
         state->paramod_count+=
            ComputeAllParamodulants(state->terms, control->ocb,
                                    tmp_copy, clause,
                                    state->processed_pos_eqns,
                                    state->tmp_store, state->freshvars,
                                    control->heuristic_parms.pm_type);

         if(control->heuristic_parms.enable_neg_unit_paramod && !ClauseIsNegative(clause))
         { /* We never need to try to overlap purely negative clauses! */
            state->paramod_count+=
               ComputeAllParamodulants(state->terms, control->ocb,
                                       tmp_copy, clause,
                                       state->processed_neg_units,
                                       state->tmp_store, state->freshvars,
                                       control->heuristic_parms.pm_type);
         }
         state->paramod_count+=
            ComputeAllParamodulants(state->terms, control->ocb,
                                    tmp_copy, clause,
                                    state->processed_non_units,
                                    state->tmp_store, state->freshvars,
                                    control->heuristic_parms.pm_type);
      }
      PERF_CTR_EXIT(ParamodTimer);
   }
}


/*-----------------------------------------------------------------------
//
// Function: eval_clause_set()
//
//   Add evaluations to all clauses in state->eval_set. Factored out
//   so that batch-processing with e.g. neural networks can be easily
//   integrated.
//
// Global Variables: -
//
// Side Effects    : -
//
/----------------------------------------------------------------------*/

void eval_clause_set(ProofState_p state, ProofControl_p control)
{
   Clause_p handle;
   assert(state);
   assert(control);

   for(handle = state->eval_store->anchor->succ;
       handle != state->eval_store->anchor;
       handle = handle->succ)
   {
      HCBClauseEvaluate(control->hcb, handle);
   }
}


/*-----------------------------------------------------------------------
//
// Function: insert_new_clauses()
//
//   Rewrite clauses in state->tmp_store, remove superfluous literals,
//   insert them into state->unprocessed. If an empty clause is
//   detected, return it, otherwise return NULL.
//
// Global Variables: -
//
// Side Effects    : As described.
//
/----------------------------------------------------------------------*/

static Clause_p insert_new_clauses(ProofState_p state, ProofControl_p control)
{
   Clause_p handle;
   long     clause_count;

   state->generated_count+=state->tmp_store->members;
   state->generated_lit_count+=state->tmp_store->literals;
   while((handle = ClauseSetExtractFirst(state->tmp_store)))
   {
      /* printf("Inserting: ");
         ClausePrint(stdout, handle, true);
         printf("\n"); */
      if(ClauseQueryProp(handle,CPIsIRVictim))
      {
         assert(ClauseQueryProp(handle, CPLimitedRW));
         ForwardModifyClause(state, control, handle,
                             control->heuristic_parms.forward_context_sr_aggressive||
                             (control->heuristic_parms.backward_context_sr&&
                              ClauseQueryProp(handle,CPIsProcessed)),
                             control->heuristic_parms.condensing_aggressive,
                             FullRewrite);
         ClauseDelProp(handle,CPIsIRVictim);
      }
      ForwardModifyClause(state, control, handle,
                          control->heuristic_parms.forward_context_sr_aggressive||
                          (control->heuristic_parms.backward_context_sr&&
                           ClauseQueryProp(handle,CPIsProcessed)),
                          control->heuristic_parms.condensing_aggressive,
                          control->heuristic_parms.forward_demod);


      if(ClauseIsTrivial(handle))
      {
         ClauseFree(handle);
         continue;
      }




      check_watchlist(&(state->wlindices), state->watchlist,
                      handle, state->archive,
                      control->heuristic_parms.watchlist_is_static);
      if(ClauseIsEmpty(handle))
      {
         return handle;
      }

      if(control->heuristic_parms.forward_subsumption_aggressive)
      {
         FVPackedClause_p pclause;

         pclause = ForwardSubsumption(state,
                                      handle,
                                      &(state->aggressive_forward_subsumed_count),
                                      true);
         if(pclause)
         {  // Not subsumed
            FVUnpackClause(pclause);
            ENSURE_NULL(pclause);
         }
         else
         {
            ClauseFree(handle);
            continue;
         }
      }

      if(control->heuristic_parms.er_aggressive &&
         control->heuristic_parms.er_varlit_destructive &&
         (clause_count =
          ClauseERNormalizeVar(state->terms,
                               handle,
                               state->tmp_store,
                               state->freshvars,
                               control->heuristic_parms.er_strong_destructive)))
      {
         state->other_redundant_count += clause_count;
         state->resolv_count += clause_count;
         state->generated_count += clause_count;
         continue;
      }
      if(control->heuristic_parms.split_aggressive &&
         (clause_count = ControlledClauseSplit(state->definition_store,
                                               handle,
                                               state->tmp_store,
                                               control->heuristic_parms.split_clauses,
                                               control->heuristic_parms.split_method,
                                               control->heuristic_parms.split_fresh_defs)))
      {
         state->generated_count += clause_count;
         continue;
      }
      state->non_trivial_generated_count++;
      ClauseDelProp(handle, CPIsOriented);
      if(!control->heuristic_parms.select_on_proc_only)
      {
         DoLiteralSelection(control, handle);
      }
      else
      {
         EqnListDelProp(handle->literals, EPIsSelected);
      }
      handle->create_date = state->proc_non_trivial_count;
      if(ProofObjectRecordsGCSelection)
      {
         ClausePushDerivation(handle, DCCnfEvalGC, NULL, NULL);
      }
//      HCBClauseEvaluate(control->hcb, handle);

      ClauseSetInsert(state->eval_store, handle);
   }
   eval_clause_set(state, control);

   while((handle = ClauseSetExtractFirst(state->eval_store)))
   {
      ClauseDelProp(handle, CPIsOriented);
      DocClauseQuoteDefault(6, handle, "eval");

      ClauseSetInsert(state->unprocessed, handle);
   }
   return NULL;
}


/*-----------------------------------------------------------------------
//
// Function: replacing_inferences()
//
//   Perform the inferences that replace a clause by another:
//   Destructive equality-resolution and/or splitting.
//
//   Returns NULL if clause was replaced, the empty clause if this
//   produced an empty clause, and the original clause otherwise
//
// Global Variables: -
//
// Side Effects    : May insert new clauses into state. May destroy
//                   pclause (in which case it gets rid of the container)
//
/----------------------------------------------------------------------*/

Clause_p replacing_inferences(ProofState_p state, ProofControl_p
                              control, FVPackedClause_p pclause)
{
   long     clause_count;
   Clause_p res = pclause->clause;

   if(problemType == PROBLEM_HO && DestructEquivalences(res, state->tmp_store, state->archive))
   {
      pclause->clause = NULL;
   }
   else
   if(control->heuristic_parms.er_varlit_destructive &&
      (clause_count =
       ClauseERNormalizeVar(state->terms,
                            pclause->clause,
                            state->tmp_store,
                            state->freshvars,
                            control->heuristic_parms.er_strong_destructive)))
   {
      state->other_redundant_count += clause_count;
      state->resolv_count += clause_count;
      pclause->clause = NULL;
   }
   else if(ControlledClauseSplit(state->definition_store,
                                 pclause->clause, state->tmp_store,
                                 control->heuristic_parms.split_clauses,
                                 control->heuristic_parms.split_method,
                                 control->heuristic_parms.split_fresh_defs))
   {
      pclause->clause = NULL;
   }

   if(!pclause->clause)
   {  /* ...then it has been destroyed by one of the above methods,
       * which may have put some clauses into tmp_store. */
      FVUnpackClause(pclause);

      res = insert_new_clauses(state, control);
   }
   return res;
}


/*-----------------------------------------------------------------------
//
// Function: cleanup_unprocessed_clauses()
//
//   Perform maintenenance operations on state->unprocessed, depending
//   on parameters in control:
//   - Remove orphaned clauses
//   - Simplify all unprocessed clauses
//   - Reweigh all unprocessed clauses
//   - Delete "bad" clauses to avoid running out of memories.
//
//   Simplification can find the empty clause, which is then
//   returned.
//
// Global Variables: -
//
// Side Effects    : As described above.
//
/----------------------------------------------------------------------*/

static Clause_p cleanup_unprocessed_clauses(ProofState_p state,
                                            ProofControl_p control)
{
   long long current_storage;
   unsigned long back_simplified;
   long tmp, tmp2;
   long target_size;
   Clause_p unsatisfiable = NULL;

   back_simplified = state->backward_subsumed_count
      +state->backward_rewritten_count;

   if((back_simplified-state->filter_orphans_base)
      > control->heuristic_parms.filter_orphans_limit)
   {
      tmp = ClauseSetDeleteOrphans(state->unprocessed);
      if(OutputLevel)
      {
         fprintf(GlobalOut,
                 "# Deleted %ld orphaned clauses (remaining: %ld)\n",
                 tmp, state->unprocessed->members);
      }
      state->other_redundant_count += tmp;
      state->filter_orphans_base = back_simplified;
   }


   if((state->processed_count-state->forward_contract_base)
      > control->heuristic_parms.forward_contract_limit)
   {
      tmp = state->unprocessed->members;
      unsatisfiable =
         ForwardContractSet(state, control,
                            state->unprocessed, false, FullRewrite,
                            &(state->other_redundant_count), true);
      if(unsatisfiable)
      {
         PStackPushP(state->extract_roots, unsatisfiable);
      }
      if(OutputLevel)
      {
         fprintf(GlobalOut,
                 "# Special forward-contraction deletes %ld clauses"
                 "(remaining: %ld) \n",
                 tmp - state->unprocessed->members,
                 state->unprocessed->members);
      }

      if(unsatisfiable)
      {
         return unsatisfiable;
      }
      state->forward_contract_base = state->processed_count;
      OUTPRINT(1, "# Reweighting unprocessed clauses...\n");
      ClauseSetReweight(control->hcb,  state->unprocessed);
   }

   current_storage  = ProofStateStorage(state);
   if(current_storage > control->heuristic_parms.delete_bad_limit)
   {
      target_size = state->unprocessed->members/2;
      tmp = ClauseSetDeleteOrphans(state->unprocessed);
      tmp2 = HCBClauseSetDeleteBadClauses(control->hcb,
                                          state->unprocessed,
                                          target_size);
      state->non_redundant_deleted += tmp;
      if(OutputLevel)
      {
         fprintf(GlobalOut,
                 "# Deleted %ld orphaned clauses and %ld bad "
                 "clauses (prover may be incomplete now)\n",
                 tmp, tmp2);
      }
      if(tmp2)
      {
         state->state_is_complete = false;
      }
      TBGCCollect(state->terms);
      current_storage = ProofStateStorage(state);
   }
   return unsatisfiable;
}

/*-----------------------------------------------------------------------
//
// Function: SATCheck()
//
//   Create ground (or pseudo-ground) instances of the clause set,
//   hand them to a SAT solver, and check then for unsatisfiability.
//
// Global Variables:
//
// Side Effects    :
//
/----------------------------------------------------------------------*/

Clause_p SATCheck(ProofState_p state, ProofControl_p control)
{
   Clause_p     empty = NULL;
   ProverResult res;
   double
      base_time,
      preproc_time = 0.0,
      enc_time     = 0.0,
      solver_time  = 0.0;

   if(control->heuristic_parms.sat_check_normalize)
   {
      //printf("# Cardinality of unprocessed: %ld\n",
      //        ClauseSetCardinality(state->unprocessed));
      base_time = GetTotalCPUTime();
      empty = ForwardContractSetReweight(state, control, state->unprocessed,
                                       false, 2,
                                       &(state->proc_trivial_count));
      // printf("# ForwardContraction done\n");
      preproc_time = (GetTotalCPUTime()-base_time);
   }
   if(!empty)
   {
      SatClauseSet_p set = SatClauseSetAlloc();

      // printf("# SatCheck() %ld, %ld..\n",
      //state->proc_non_trivial_count,
      //ProofStateCardinality(state));

      base_time = GetTotalCPUTime();
      SatClauseSetImportProofState(set, state,
                                   control->heuristic_parms.sat_check_grounding,
                                   control->heuristic_parms.sat_check_normconst);

      enc_time = (GetTotalCPUTime()-base_time);
      //printf("# SatCheck()..imported\n");

      base_time = GetTotalCPUTime();
      res = SatClauseSetCheckUnsat(set, &empty, control->solver,
                                   control->heuristic_parms.sat_check_decision_limit);
      ProofControlResetSATSolver(control);
      solver_time = (GetTotalCPUTime()-base_time);
      state->satcheck_count++;

      state->satcheck_preproc_time  += preproc_time;
      state->satcheck_encoding_time += enc_time;
      state->satcheck_solver_time   += solver_time;
      if(res == PRUnsatisfiable)
      {
         state->satcheck_success++;
         state->satcheck_full_size = SatClauseSetCardinality(set);
         state->satcheck_actual_size = SatClauseSetNonPureCardinality(set);
         state->satcheck_core_size = SatClauseSetCoreSize(set);

         state->satcheck_preproc_stime  += preproc_time;
         state->satcheck_encoding_stime += enc_time;
         state->satcheck_solver_stime   += solver_time;
      }
      else if(res == PRSatisfiable)
      {
         state->satcheck_satisfiable++;
      }
      SatClauseSetFree(set);
   }

   return empty;
}

#ifdef PRINT_SHARING

/*-----------------------------------------------------------------------
//
// Function: print_sharing_factor()
//
//   Determine the sharing factor and print it. Potentially expensive,
//   only useful for manual analysis.
//
// Global Variables:
//
// Side Effects    :
//
/----------------------------------------------------------------------*/

static void print_sharing_factor(ProofState_p state)
{
   long shared, unshared;
   shared = TBTermNodes(state->terms);
   unshared = ClauseSetGetTermNodes(state->tmp_store)+
      ClauseSetGetTermNodes(state->processed_pos_rules)+
      ClauseSetGetTermNodes(state->processed_pos_eqns)+
      ClauseSetGetTermNodes(state->processed_neg_units)+
      ClauseSetGetTermNodes(state->processed_non_units)+
      ClauseSetGetTermNodes(state->unprocessed);

   fprintf(GlobalOut, "\n#        GClauses   NClauses    NShared  "
           "NUnshared    TShared  TUnshared TSharinF   \n");
   fprintf(GlobalOut, "# grep %10ld %10ld %10ld %10ld %10ld %10ld %10.3f\n",
           state->generated_count - state->backward_rewritten_count,
           state->tmp_store->members,
           ClauseSetGetSharedTermNodes(state->tmp_store),
           ClauseSetGetTermNodes(state->tmp_store),
           shared,
           unshared,
           (float)unshared/shared);
}
#endif


#ifdef PRINT_RW_STATE

/*-----------------------------------------------------------------------
//
// Function: print_rw_state()
//
//   Print the system (R,E,NEW), e.g. the two types of demodulators
//   and the newly generated clauses.
//
// Global Variables: -
//
// Side Effects    : Output
//
/----------------------------------------------------------------------*/

void print_rw_state(ProofState_p state)
{
   char prefix[20];

   sprintf(prefix, "RWD%09ld_R: ",state->proc_non_trivial_count);
   ClauseSetPrintPrefix(GlobalOut,prefix,state->processed_pos_rules);
   sprintf(prefix, "RWD%09ld_E: ",state->proc_non_trivial_count);
   ClauseSetPrintPrefix(GlobalOut,prefix,state->processed_pos_eqns);
   sprintf(prefix, "RWD%09ld_N: ",state->proc_non_trivial_count);
   ClauseSetPrintPrefix(GlobalOut,prefix,state->tmp_store);
}

#endif





/*---------------------------------------------------------------------*/
/*                         Exported Functions                          */
/*---------------------------------------------------------------------*/



/*-----------------------------------------------------------------------
//
// Function: ProofControlInit()
//
//   Initialize a proof control cell for a given proof state (with at
//   least axioms and signature) and a set of parameters
//   describing the ordering and heuristics.
//
// Global Variables: -
//
// Side Effects    : Sets specs.
//
/----------------------------------------------------------------------*/

void ProofControlInit(ProofState_p state, ProofControl_p control,
                      HeuristicParms_p params, FVIndexParms_p fvi_params,
                      PStack_p wfcb_defs, PStack_p hcb_defs)
{
   PStackPointer sp;
   Scanner_p in;

   assert(control && control->wfcbs);
   assert(state && state->axioms && state->signature);
   assert(params);
   assert(!control->ocb);
   assert(!control->hcb);

   SpecFeaturesCompute(&(control->problem_specs),
                       state->axioms,state->signature);

   control->ocb = TOSelectOrdering(state, params,
                                   &(control->problem_specs));

   in = CreateScanner(StreamTypeInternalString,
                      DefaultWeightFunctions,
                      true, NULL, true);
   WeightFunDefListParse(control->wfcbs, in, control->ocb, state);
   DestroyScanner(in);

   for(sp = 0; sp < PStackGetSP(wfcb_defs); sp++)
   {
      in = CreateScanner(StreamTypeOptionString,
                         PStackElementP(wfcb_defs, sp) ,
                         true, NULL, true);
      WeightFunDefListParse(control->wfcbs, in, control->ocb, state);
      DestroyScanner(in);
   }
   in = CreateScanner(StreamTypeInternalString,
                      DefaultHeuristics,
                      true, NULL, true);
   HeuristicDefListParse(control->hcbs, in, control->wfcbs,
                         control->ocb, state);
   AcceptInpTok(in, Fullstop);
   DestroyScanner(in);
   if(!PStackEmpty(hcb_defs))
   {
      params->heuristic_def = SecureStrdup(PStackTopP(hcb_defs));
   }
   for(sp = 0; sp < PStackGetSP(hcb_defs); sp++)
   {
      in = CreateScanner(StreamTypeOptionString,
                         PStackElementP(hcb_defs, sp) ,
                         true, NULL, true);
      HeuristicDefListParse(control->hcbs, in, control->wfcbs,
                            control->ocb, state);
      DestroyScanner(in);
   }
   control->heuristic_parms     = *params;

   control->hcb = GetHeuristic(params->heuristic_name,
                               state,
                               control,
                               params);
   control->fvi_parms           = *fvi_params;
   if(!control->heuristic_parms.split_clauses)
   {
      control->fvi_parms.symbol_slack = 0;
   }
   *params = control->heuristic_parms;
}


/*-----------------------------------------------------------------------
//
// Function: ProofStateResetProcessedSet()
//
//   Move all clauses from set into state->unprocessed.
//
// Global Variables: -
//
// Side Effects    : -
//
/----------------------------------------------------------------------*/

void ProofStateResetProcessedSet(ProofState_p state,
                                 ProofControl_p control,
                                 ClauseSet_p set)
{
   Clause_p handle;
   Clause_p tmpclause;

   while((handle = ClauseSetExtractFirst(set)))
   {
      if(ClauseQueryProp(handle, CPIsGlobalIndexed))
      {
         GlobalIndicesDeleteClause(&(state->gindices), handle);
      }
      if(ProofObjectRecordsGCSelection)
      {
         ClausePushDerivation(handle, DCCnfEvalGC, NULL, NULL);
      }
      tmpclause = ClauseFlatCopy(handle);
      ClausePushDerivation(tmpclause, DCCnfQuote, handle, NULL);
      ClauseSetInsert(state->archive, handle);
      handle = tmpclause;
      HCBClauseEvaluate(control->hcb, handle);
      ClauseDelProp(handle, CPIsOriented);
      DocClauseQuoteDefault(6, handle, "move_eval");

      if(control->heuristic_parms.prefer_initial_clauses)
      {
         EvalListChangePriority(handle->evaluations, -PrioLargestReasonable);
      }
      ClauseSetInsert(state->unprocessed, handle);
   }
}


/*-----------------------------------------------------------------------
//
// Function: ProofStateResetProcessed()
//
//   Move all clauses from the processed clause sets to unprocessed.
//
// Global Variables: -
//
// Side Effects    : -
//
/----------------------------------------------------------------------*/

void ProofStateResetProcessed(ProofState_p state, ProofControl_p control)
{
   ProofStateResetProcessedSet(state, control, state->processed_pos_rules);
   ProofStateResetProcessedSet(state, control, state->processed_pos_eqns);
   ProofStateResetProcessedSet(state, control, state->processed_neg_units);
   ProofStateResetProcessedSet(state, control, state->processed_non_units);
}


/*-----------------------------------------------------------------------
//
// Function: fvi_param_init()
//
//   Initialize the parameters for all feature vector indices in
//   state.
//
// Global Variables: -
//
// Side Effects    : -
//
/----------------------------------------------------------------------*/

void fvi_param_init(ProofState_p state, ProofControl_p control)
{
   long symbols;
   PermVector_p perm;
   FVCollect_p  cspec;

   state->fvi_initialized = true;
   state->original_symbols = state->signature->f_count;

   symbols = MIN(state->original_symbols+control->fvi_parms.symbol_slack,
                 control->fvi_parms.max_symbols);

   // printf("### Symbols: %ld\n", symbols);
   switch(control->fvi_parms.cspec.features)
   {
   case FVIBillFeatures:
         cspec = BillFeaturesCollectAlloc(state->signature, symbols*2+2);
         break;
   case FVIBillPlusFeatures:
         cspec = BillPlusFeaturesCollectAlloc(state->signature, symbols*2+4);
         break;
   case FVIACFold:
         // printf("# FVIACFold\n");
         cspec = FVCollectAlloc(FVICollectFeatures,
                                true,
                                0,
                                symbols*2+2,
                                2,
                                0,
                                symbols,
                                symbols+2,
                                0,
                                symbols,
                                0,0,0,
                                0,0,0);
         break;
   case FVIACStagger:
         cspec = FVCollectAlloc(FVICollectFeatures,
                                true,
                                0,
                                symbols*2+2,
                                2,
                                0,
                                2*symbols,
                                2,
                                2+symbols,
                                2*symbols,
                                0,0,0,
                                0,0,0);
         break;
   case FVICollectFeatures:
         cspec = FVCollectAlloc(control->fvi_parms.cspec.features,
                                control->fvi_parms.cspec.use_litcount,
                                control->fvi_parms.cspec.ass_vec_len,
                                symbols,
                                control->fvi_parms.cspec.pos_count_base,
                                control->fvi_parms.cspec.pos_count_offset,
                                control->fvi_parms.cspec.pos_count_mod,
                                control->fvi_parms.cspec.neg_count_base,
                                control->fvi_parms.cspec.neg_count_offset,
                                control->fvi_parms.cspec.neg_count_mod,
                                control->fvi_parms.cspec.pos_depth_base,
                                control->fvi_parms.cspec.pos_depth_offset,
                                control->fvi_parms.cspec.pos_depth_mod,
                                control->fvi_parms.cspec.neg_depth_base,
                                control->fvi_parms.cspec.neg_depth_offset,
                                control->fvi_parms.cspec.neg_depth_mod);

         break;
   default:
         cspec = FVCollectAlloc(control->fvi_parms.cspec.features,
                                0,
                                0,
                                0,
                                0, 0, 0,
                                0, 0, 0,
                                0, 0, 0,
                                0, 0, 0);
         break;
   }
   cspec->max_symbols=symbols;
   state->fvi_cspec = cspec;

   perm = PermVectorCompute(state->axioms,
                            cspec,
                            control->fvi_parms.eliminate_uninformative);
   if(control->fvi_parms.cspec.features != FVINoFeatures)
   {
      state->processed_non_units->fvindex =
         FVIAnchorAlloc(cspec, PermVectorCopy(perm));
      state->processed_pos_rules->fvindex =
         FVIAnchorAlloc(cspec, PermVectorCopy(perm));
      state->processed_pos_eqns->fvindex =
         FVIAnchorAlloc(cspec, PermVectorCopy(perm));
      state->processed_neg_units->fvindex =
         FVIAnchorAlloc(cspec, PermVectorCopy(perm));
      if(state->watchlist)
      {
         state->watchlist->fvindex =
            FVIAnchorAlloc(cspec, PermVectorCopy(perm));
         //ClauseSetNewTerms(state->watchlist, state->terms);
      }
   }
   state->def_store_cspec = FVCollectAlloc(FVICollectFeatures,
                                           true,
                                           0,
                                           symbols*2+2,
                                           2,
                                           0,
                                           symbols,
                                           symbols+2,
                                           0,
                                           symbols,
                                           0,0,0,
                                           0,0,0);
   state->definition_store->def_clauses->fvindex =
      FVIAnchorAlloc(state->def_store_cspec, perm);
}



/*-----------------------------------------------------------------------
//
// Function: ProofStateInit()
//
//   Given a proof state with axioms and a heuristic parameter
//   description, initialize the ProofStateCell, i.e. generate the
//   HCB, the ordering, and evaluate the axioms and put them in the
//   unprocessed list.
//
// Global Variables:
//
// Side Effects    :
//
/----------------------------------------------------------------------*/

void ProofStateInit(ProofState_p state, ProofControl_p control)
{
   Clause_p handle, new;
   HCB_p    tmphcb;
   PStack_p traverse;
   Eval_p   cell;

   OUTPRINT(1, "# Initializing proof state\n");

   assert(ClauseSetEmpty(state->processed_pos_rules));
   assert(ClauseSetEmpty(state->processed_pos_eqns));
   assert(ClauseSetEmpty(state->processed_neg_units));
   assert(ClauseSetEmpty(state->processed_non_units));

   if(!state->fvi_initialized)
   {
      fvi_param_init(state, control);
   }
   ProofStateInitWatchlist(state, control->ocb);

   tmphcb = GetHeuristic("Uniq", state, control, &(control->heuristic_parms));
   assert(tmphcb);
   ClauseSetReweight(tmphcb, state->axioms);

   traverse =
      EvalTreeTraverseInit(PDArrayElementP(state->axioms->eval_indices,0),0);

   while((cell = EvalTreeTraverseNext(traverse, 0)))
   {
      handle = cell->object;
      new = ClauseCopy(handle, state->terms);

      ClauseSetProp(new, CPInitial);
      check_watchlist(&(state->wlindices), state->watchlist,
                      new, state->archive,
                      control->heuristic_parms.watchlist_is_static);
      HCBClauseEvaluate(control->hcb, new);
      DocClauseQuoteDefault(6, new, "eval");
      ClausePushDerivation(new, DCCnfQuote, handle, NULL);
      if(ProofObjectRecordsGCSelection)
      {
         ClausePushDerivation(new, DCCnfEvalGC, NULL, NULL);
      }
      if(control->heuristic_parms.prefer_initial_clauses)
      {
         EvalListChangePriority(new->evaluations, -PrioLargestReasonable);
      }
      ClauseSetInsert(state->unprocessed, new);
   }
   OUTPRINT(1, "# Initializing proof state (3)\n");
   ClauseSetMarkSOS(state->unprocessed, control->heuristic_parms.use_tptp_sos);
   // printf("Before EvalTreeTraverseExit\n");
   EvalTreeTraverseExit(traverse);

   if(control->heuristic_parms.ac_handling!=NoACHandling)
   {
      if(OutputLevel)
      {
         fprintf(GlobalOut, "# Scanning for AC axioms\n");
      }
      control->ac_handling_active = ClauseSetScanAC(state->signature,
                                                    state->unprocessed);
      if(OutputLevel)
      {
         SigPrintACStatus(GlobalOut, state->signature);
         if(control->ac_handling_active)
         {
            fprintf(GlobalOut, "# AC handling enabled\n");
         }
      }
   }

   GlobalIndicesInit(&(state->gindices),
                     state->signature,
                     control->heuristic_parms.rw_bw_index_type,
                     control->heuristic_parms.pm_from_index_type,
                     control->heuristic_parms.pm_into_index_type,
                     control->heuristic_parms.ext_sup_max_depth);

}


/*-----------------------------------------------------------------------
//
// Function: ProcessClause()
//
//   Select an unprocessed clause, process it. Return pointer to empty
//   clause if it can be derived, NULL otherwise. This is the core of
//   the main proof procedure.
//
// Global Variables: -
//
// Side Effects    : Everything ;-)
//
/----------------------------------------------------------------------*/

static Clause_p trailClause; // HACK - avoid hassle adding param
static void writeTrailInput(ProofState_p state, ProofControl_p control);
static void maybeOpenTrailPipes(FILE **trailInputp, FILE **trailOutputp);
static Clause_p readTrailOutput(ProofState_p state, FILE *trailInput, FILE *trailOutput);
static void reopenTrailPipes(FILE *trailInput, FILE *trailOutput);
static Clause_p maybe_select_single_lit(ProofState_p state);
static int always_select_single_lit_action=0;
Clause_p ProcessClause(ProofState_p state, ProofControl_p control,
                       long answer_limit)
{
   Clause_p         clause, resclause, tmp_copy, empty, arch_copy = NULL;
   FVPackedClause_p pclause;
   SysDate          clausedate;

   if (trailClause != NULL) {
      clause = trailClause;
   } else
   clause = control->hcb->hcb_select(control->hcb,
                                     state->unprocessed);
   if(!clause)
   {
      return NULL;
   }
   //EvalListPrintComment(GlobalOut, clause->evaluations); printf("\n");
   if(OutputLevel==1)
   {
      putc('#', GlobalOut);
   }
   assert(clause);

   ClauseSetExtractEntry(clause);
   ClauseRemoveEvaluations(clause);
   // Orphans have been excluded during selection now

   ClauseSetProp(clause, CPIsProcessed);
   state->processed_count++;

   assert(!ClauseQueryProp(clause, CPIsIRVictim));

   if(ProofObjectRecordsGCSelection)
   {
      arch_copy = ClauseArchiveCopy(state->archive, clause);
   }

   if(!(pclause = ForwardContractClause(state, control,
                                        clause, true,
                                        control->heuristic_parms.forward_context_sr,
                                        control->heuristic_parms.condensing,
                                        FullRewrite)))
   {
      if(arch_copy)
      {
         ClauseSetDeleteEntry(arch_copy);
      }
      return NULL;
   }

   if(ClauseIsSemFalse(pclause->clause))
   {
      state->answer_count ++;
      ClausePrintAnswer(GlobalOut, pclause->clause, state);
      PStackPushP(state->extract_roots, pclause->clause);
      if(ClauseIsEmpty(pclause->clause)||
         state->answer_count>=answer_limit)
      {
         clause = FVUnpackClause(pclause);
         ClauseEvaluateAnswerLits(clause);
         return clause;
      }
   }
   assert(ClauseIsSubsumeOrdered(pclause->clause));
   check_ac_status(state, control, pclause->clause);

   document_processing(pclause->clause);
   state->proc_non_trivial_count++;

   resclause = replacing_inferences(state, control, pclause);
   if(!resclause || ClauseIsEmpty(resclause))
   {
      if(resclause)
      {
         PStackPushP(state->extract_roots, resclause);
      }
      return resclause;
   }

   check_watchlist(&(state->wlindices), state->watchlist,
                      pclause->clause, state->archive,
                      control->heuristic_parms.watchlist_is_static);

   /* Now on to backward simplification. */
   clausedate = ClauseSetListGetMaxDate(state->demods, FullRewrite);

   eliminate_backward_rewritten_clauses(state, control, pclause->clause, &clausedate);
   eliminate_backward_subsumed_clauses(state, pclause);
   eliminate_unit_simplified_clauses(state, pclause->clause);
   eliminate_context_sr_clauses(state, control, pclause->clause);
   ClauseSetSetProp(state->tmp_store, CPIsIRVictim);

   clause = pclause->clause;

   ClauseNormalizeVars(clause, state->freshvars);
   tmp_copy = ClauseCopyDisjoint(clause);
   tmp_copy->ident = clause->ident;

   clause->date = clausedate;
   ClauseSetProp(clause, CPLimitedRW);

   if(ClauseIsDemodulator(clause))
   {
      extern int processed_demodulator;
      processed_demodulator=1;
      assert(clause->neg_lit_no == 0);
      if(EqnIsOriented(clause->literals))
      {
         TermCellSetProp(clause->literals->lterm, TPIsRewritable);
         state->processed_pos_rules->date = clausedate;
         ClauseSetIndexedInsert(state->processed_pos_rules, pclause);
      }
      else
      {
         state->processed_pos_eqns->date = clausedate;
         ClauseSetIndexedInsert(state->processed_pos_eqns, pclause);
      }
   }
   else if(ClauseLiteralNumber(clause) == 1)
   {
      assert(clause->neg_lit_no == 1);
      ClauseSetIndexedInsert(state->processed_neg_units, pclause);
   }
   else
   {
      ClauseSetIndexedInsert(state->processed_non_units, pclause);
   }
   GlobalIndicesInsertClause(&(state->gindices), clause);

   FVUnpackClause(pclause);
   ENSURE_NULL(pclause);
   if(state->watchlist && control->heuristic_parms.watchlist_simplify)
   {
      simplify_watchlist(state, control, clause);
   }
   if(control->heuristic_parms.selection_strategy != SelectNoGeneration)
   {
      generate_new_clauses(state, control, clause, tmp_copy);
   }
   ClauseFree(tmp_copy);
   if(TermCellStoreNodes(&(state->tmp_terms->term_store))>TMPBANK_GC_LIMIT)
   {
      TBGCSweep(state->tmp_terms);
   }
#ifdef PRINT_SHARING
   print_sharing_factor(state);
#endif
#ifdef PRINT_RW_STATE
   print_rw_state(state);
#endif
   if(control->heuristic_parms.detsort_tmpset)
   {
      ClauseSetSort(state->tmp_store, ClauseCmpByStructWeight);
   }
   if((empty = insert_new_clauses(state, control)))
   {
      PStackPushP(state->extract_roots, empty);
      return empty;
   }
   return NULL;
}


/*-----------------------------------------------------------------------
//
// Function:  Saturate()
//
//   Process clauses until either the empty clause has been derived, a
//   specified number of clauses has been processed, or the clause set
//   is saturated. Return empty clause (if found) or NULL.
//
// Global Variables: -
//
// Side Effects    : Modifies state.
//
/----------------------------------------------------------------------*/

Clause_p Saturate(ProofState_p state, ProofControl_p control, long
                  step_limit, long proc_limit, long unproc_limit, long
                  total_limit, long generated_limit, long tb_insert_limit,
                  long answer_limit)
{
   Clause_p unsatisfiable = NULL;
   long
      count = 0,
      sat_check_size_limit = control->heuristic_parms.sat_check_size_limit,
      sat_check_step_limit = control->heuristic_parms.sat_check_step_limit,
      sat_check_ttinsert_limit = control->heuristic_parms.sat_check_ttinsert_limit;



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
       printf("Saturate - presumably presaturating - skipping since SKIP_PRESATURATE\n");
       return 0;
     }
   }
//     fprintf(trailInput, "%d\n", presaturating);
//     fflush(trailInput);
   if (presaturating) {
       // untested, never seen so far
       printf("Saturate - presumably presaturating\n");
   } else
       printf("Saturate - presumably NOT presaturating\n");
   // END TRAIL MODS
   while(!TimeIsUp &&
         !ClauseSetEmpty(state->unprocessed) &&
         step_limit   > count &&
         proc_limit   > ProofStateProcCardinality(state) &&
         unproc_limit > ProofStateUnprocCardinality(state) &&
         total_limit  > ProofStateCardinality(state) &&
         generated_limit > (state->generated_count -
                            state->backward_rewritten_count)&&
         tb_insert_limit > state->terms->insertions &&
         (!state->watchlist||!ClauseSetEmpty(state->watchlist)))
   {
      count++;
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
      unsatisfiable = ProcessClause(state, control, answer_limit);
      if(unsatisfiable)
      {
         break;
      }
      unsatisfiable = cleanup_unprocessed_clauses(state, control);
      if(unsatisfiable)
      {
         break;
      }
      if(control->heuristic_parms.sat_check_grounding != GMNoGrounding)
      {
         if(ProofStateCardinality(state) >= sat_check_size_limit)
         {
            unsatisfiable = SATCheck(state, control);
            while(sat_check_size_limit <= ProofStateCardinality(state))
            {
               sat_check_size_limit += control->heuristic_parms.sat_check_size_limit;
            }
         }
         else if(state->proc_non_trivial_count >= sat_check_step_limit)
         {
            unsatisfiable = SATCheck(state, control);
            sat_check_step_limit += control->heuristic_parms.sat_check_step_limit;
         }
         else if( state->terms->insertions >= sat_check_ttinsert_limit)
         {
            unsatisfiable = SATCheck(state, control);
            sat_check_ttinsert_limit *=2;
         }
         if(unsatisfiable)
         {
            PStackPushP(state->extract_roots, unsatisfiable);
            break;
         }
      }
   }
   return unsatisfiable;
}




/*---------------------------------------------------------------------*/
/*                        End of File                                  */
/*---------------------------------------------------------------------*/
static int EPROVER_PROTOCOL=11;
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

  fwrite(&naa, sizeof(int), 1, out);
  fwrite(aabuf, sizeof(int), naa, out); 
  fflush(out);
  printf("wrote newunprocessed %d\n", naa); fflush(stdout);

  fwrite(&npc, sizeof(int), 1, out);
  fwrite(pcbuf, sizeof(int), npc, out); 
  fflush(out);
  printf("wrote newprocessed %d\n", npc); fflush(stdout);

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

static int stopping=0;
// you MUST write these IN THE SAME ORDER as writeTrailInput
// ... unless we were told to stop, in which case no more input is expected.
// by not exiting immediately, we get the E summary.
void writeDummyTrailInput() {
  FILE *out;
  // NUTS - repeated from above
  char *trailOutputName = "Trail2E.pipe";
  int trailOutputExists = (access(trailOutputName, F_OK ) == 0);

  if (trailOutputExists) {
    if (stopping) {
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
              stopping=1; // nuts
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
