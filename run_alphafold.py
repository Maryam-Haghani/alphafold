# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Full AlphaFold protein structure prediction script."""
import enum
import json
import os
import pathlib
import pickle
import random
import shutil
import sys
import time
from typing import Any, Dict, Mapping, Union
from absl import app
from absl import flags
from absl import logging
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.data import pipeline
from alphafold.data import pipeline_multimer
from alphafold.data import templates
from alphafold.data.tools import hhsearch
from alphafold.data.tools import hmmsearch
from alphafold.model import config
from alphafold.model import data
from alphafold.model import model
from alphafold.relax import relax
import jax.numpy as jnp
import numpy as np

# Internal import (7716).

logging.set_verbosity(logging.INFO)


@enum.unique
class ModelsToRelax(enum.Enum):
  ALL = 0
  BEST = 1
  NONE = 2
  TOP5 = 3

flags.DEFINE_list(
    'fasta_paths', None, 'Paths to FASTA files, each containing a prediction '
    'target that will be folded one after another. If a FASTA file contains '
    'multiple sequences, then it will be folded as a multimer. Paths should be '
    'separated by commas. All FASTA paths must have a unique basename as the '
    'basename is used to name the output directories for each prediction.')

data_dir = "/projects/protein-structure-prediction/AlphaFold-Data/"

DOWNLOAD_DIR= data_dir
uniprot_database_path = os.path.join(
    DOWNLOAD_DIR, 'uniprot',    'uniprot.fasta')
uniref90_database_path = os.path.join(
    DOWNLOAD_DIR, 'uniref90', 'uniref90.fasta')
mgnify_database_path = os.path.join(
    DOWNLOAD_DIR, 'mgnify', 'mgy_clusters_2022_05.fa')
small_bfd_database_path = os.path.join(
    DOWNLOAD_DIR, 'small_bfd',    'bfd-first_non_consensus_sequences.fasta')
bfd_database_path = os.path.join(
    DOWNLOAD_DIR, 'bfd',    'bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt')
uniclust30_database_path = os.path.join(DOWNLOAD_DIR, 'uniref30-2023', 'UniRef30_2023_02')
# DOWNLOAD_DIR, 'uniref30', 'UniRef30_2021_03')
pdb70_database_path = os.path.join(DOWNLOAD_DIR, 'pdb70', 'pdb70')
template_mmcif_dir = os.path.join(DOWNLOAD_DIR, 'pdb_mmcif', 'mmcif_files')
obsolete_pdbs_path = os.path.join(DOWNLOAD_DIR, 'pdb_mmcif', 'obsolete.dat')
pdb_seqres_database_path=os.path.join(DOWNLOAD_DIR, 'pdb_seqres', 'pdb_seqres.txt')
binary_path = '/home/haghani/.conda/envs/alphafold/bin/'

flags.DEFINE_string('data_dir', None, 'Path to directory of supporting data.')
flags.DEFINE_string('output_dir', None, 'Path to a directory that will '
                    'store the results.')
flags.DEFINE_string('prediction_dir', None, 'Path to a directory that will '
                    'store the prediction results.')
flags.DEFINE_string('jackhmmer_binary_path', binary_path+'jackhmmer',
                    'Path to the JackHMMER executable.')
flags.DEFINE_string('hhblits_binary_path', binary_path+'hhblits',
                    'Path to the HHblits executable.')
flags.DEFINE_string('hhsearch_binary_path', binary_path+'hhsearch',
                    'Path to the HHsearch executable.')
flags.DEFINE_string('hmmsearch_binary_path', binary_path+'hmmsearch',
                    'Path to the hmmsearch executable.')
flags.DEFINE_string('hmmbuild_binary_path', binary_path+'hmmbuild',
                    'Path to the hmmbuild executable.')
flags.DEFINE_string('kalign_binary_path', binary_path+'kalign',
                    'Path to the Kalign executable.')
flags.DEFINE_string('uniref90_database_path', uniref90_database_path, 'Path to the Uniref90 '
                    'database for use by JackHMMER.')
flags.DEFINE_string('mgnify_database_path', mgnify_database_path, 'Path to the MGnify '
                    'database for use by JackHMMER.')
flags.DEFINE_string('bfd_database_path', bfd_database_path , 'Path to the BFD '
                    'database for use by HHblits.')
flags.DEFINE_string('small_bfd_database_path', None, 'Path to the small '
                    'version of BFD used with the "reduced_dbs" preset.')
flags.DEFINE_string('uniref30_database_path', uniclust30_database_path, 'Path to the UniRef30 '
                    'database for use by HHblits.')
flags.DEFINE_string('uniprot_database_path', uniprot_database_path , 'Path to the Uniprot '
                    'database for use by JackHMMer.')
flags.DEFINE_string('pdb70_database_path', pdb70_database_path, 'Path to the PDB70 ' #pdb70_database_path
                    'database for use by HHsearch.')
flags.DEFINE_string('pdb_seqres_database_path', pdb_seqres_database_path, 'Path to the PDB '
                    'seqres database for use by hmmsearch.')
flags.DEFINE_string('template_mmcif_dir', template_mmcif_dir, 'Path to a directory with '
                    'template mmCIF structures, each named <pdb_id>.cif')
flags.DEFINE_string('max_template_date', None, 'Maximum template release date '
                    'to consider. Important if folding historical test sets.')
flags.DEFINE_string('obsolete_pdbs_path',obsolete_pdbs_path , 'Path to file containing a '
                    'mapping from obsolete PDB IDs to the PDB IDs of their '
                    'replacements.')
flags.DEFINE_enum('db_preset', 'full_dbs',
                  ['full_dbs', 'reduced_dbs'],
                  'Choose preset MSA database configuration - '
                  'smaller genetic database config (reduced_dbs) or '
                  'full genetic database config  (full_dbs)')
flags.DEFINE_enum('model_preset', 'multimer',
                  ['monomer', 'monomer_casp14', 'monomer_ptm', 'multimer'],
                  'Choose preset model configuration - the monomer model, '
                  'the monomer model with extra ensembling, monomer model with '
                  'pTM head, or multimer model')
flags.DEFINE_boolean('benchmark', False, 'Run multiple JAX model evaluations '
                     'to obtain a timing that excludes the compilation time, '
                     'which should be more indicative of the time required for '
                     'inferencing many proteins.')
flags.DEFINE_integer('random_seed', None, 'The random seed for the data '
                     'pipeline. By default, this is randomly generated. Note '
                     'that even if this is set, Alphafold may still not be '
                     'deterministic, because processes like GPU inference are '
                     'nondeterministic.')
flags.DEFINE_integer('num_multimer_predictions_per_model', 5, 'How many '
                     'predictions (each with a different random seed) will be '
                     'generated per model. E.g. if this is 2 and there are 5 '
                     'models then there will be 10 predictions per input. '
                     'Note: this FLAG only applies if model_preset=multimer')
flags.DEFINE_boolean('use_templates', True, 'Whether to use templates for prediction or not.')
flags.DEFINE_boolean('only_features', False, 'Not predict structure, only get features')
flags.DEFINE_boolean('no_MSA', False, 'Whether to use MSAs for prediction or not (for multimers).')
flags.DEFINE_boolean('save_chain_msa', False, 'Whether to save final MSA file for each of the chains.'
                                              ' Does not include uniprot MSA features for multimers')
flags.DEFINE_boolean('use_precomputed_msas', False, 'Whether to read MSAs that '
                     'have been written to disk instead of running the MSA '
                     'tools. The MSA files are looked up in the output '
                     'directory, so it must stay the same between multiple '
                     'runs that are to reuse the MSAs. WARNING: This will not '
                     'check if the sequence, database or configuration have '
                     'changed.')
flags.DEFINE_boolean('use_precomputed_msas_from_dir', False, "Whether to read each chain MSAs in MSAs 'directory' of"
                     "msa_output_dir instead of computing MSA based of bfd, mgnify and uniref90 datasets"
                     "can be multiple msas in he MSAs directory, but should be either sto or a3m format")

flags.DEFINE_boolean('use_pairing', True, 'Whether to pair MSAs'
                     'of individual monomers in multimers or not')
flags.DEFINE_boolean('use_precomputed_paired_msa', False, 'Whether to read paired MSA'
                     'instead of computing that based of uniprot')
flags.DEFINE_string('precomputed_paired_msa_file', '', 'path of paired msa file which is in sto format')
flags.DEFINE_boolean('save_multimer_msa', False, 'Whether to save final MSA file for multimers')

flags.DEFINE_enum_class('models_to_relax', ModelsToRelax.TOP5, ModelsToRelax,
                        'The models to run the final relaxation step on. '
                        'If `all`, all models are relaxed, which may be time '
                        'consuming. If `best`, only the most confident model '
                        'is relaxed. If `none`, relaxation is not run. Turning '
                        'off relaxation might result in predictions with '
                        'distracting stereochemical violations but might help '
                        'in case you are having issues with the relaxation '
                        'stage.')
flags.DEFINE_boolean('use_gpu_relax', False, 'Whether to relax on GPU. '
                     'Relax on GPU can be much faster than CPU, so it is '
                     'recommended to enable if possible. GPUs must be available'
                     ' if this setting is enabled.')

FLAGS = flags.FLAGS

MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3


def _check_flag(flag_name: str,
                other_flag_name: str,
                should_be_set: bool):
  if should_be_set != bool(FLAGS[flag_name].value):
    verb = 'be' if should_be_set else 'not be'
    raise ValueError(f'{flag_name} must {verb} set when running with '
                     f'"--{other_flag_name}={FLAGS[other_flag_name].value}".')


def _jnp_to_np(output: Dict[str, Any]) -> Dict[str, Any]:
  """Recursively changes jax arrays to numpy arrays."""
  for k, v in output.items():
    if isinstance(v, dict):
      output[k] = _jnp_to_np(v)
    elif isinstance(v, jnp.ndarray):
      output[k] = np.array(v)
  return output

def predict_structure(
    fasta_path: str,
    fasta_name: str,
    output_dir_base: str,
    prediction_dir: str,
    data_pipeline: Union[pipeline.DataPipeline, pipeline_multimer.DataPipeline],
    model_runners: Dict[str, model.RunModel],
    amber_relaxer: relax.AmberRelaxation,
    benchmark: bool,
    random_seed: int,
    models_to_relax: ModelsToRelax):
  """Predicts structure using AlphaFold for the given sequence."""
  logging.info('Predicting %s', fasta_name)
  timings = {}
  output_dir = os.path.join(output_dir_base, fasta_name)
  os.makedirs(output_dir, exist_ok=True)

  if prediction_dir is not None:
      current_prediction_dir = os.path.join(output_dir, prediction_dir)
  else:
      current_prediction_dir = output_dir
  os.makedirs(current_prediction_dir, exist_ok=True)

  msa_output_dir = os.path.join(output_dir, 'msas')
  os.makedirs(msa_output_dir, exist_ok=True)
  # Get features.
  t_0 = time.time()

  feature_dict = data_pipeline.process(
      input_fasta_path=fasta_path,
      msa_output_dir=msa_output_dir
  )
  timings['features'] = time.time() - t_0

  # Write out features as a pickled dictionary.
  features_output_path = os.path.join(current_prediction_dir, 'features.pkl')
  with open(features_output_path, 'wb') as f:
    pickle.dump(feature_dict, f, protocol=4)

  if FLAGS.only_features:
      return

  unrelaxed_pdbs = {}
  unrelaxed_proteins = {}
  relaxed_pdbs = {}
  relax_metrics = {}
  ranking_confidences = {}

  # Run the models.
  num_models = len(model_runners)
  for model_index, (model_name, model_runner) in enumerate(model_runners.items()):
    logging.info('Running model %s on %s', model_name, fasta_name)
    t_0 = time.time()
    model_random_seed = model_index + random_seed * num_models
    processed_feature_dict = model_runner.process_features(
        feature_dict, random_seed=model_random_seed)
    timings[f'process_features_{model_name}'] = time.time() - t_0

    t_0 = time.time()
    prediction_result = model_runner.predict(processed_feature_dict, random_seed=model_random_seed)
    t_diff = time.time() - t_0
    timings[f'predict_and_compile_{model_name}'] = t_diff
    logging.info('Total JAX model %s on %s predict time (includes compilation time, see --benchmark): %.1fs',
        model_name, fasta_name, t_diff)

    if benchmark:
      t_0 = time.time()
      model_runner.predict(processed_feature_dict, random_seed=model_random_seed)
      t_diff = time.time() - t_0
      timings[f'predict_benchmark_{model_name}'] = t_diff
      logging.info(
          'Total JAX model %s on %s predict time (excludes compilation time): %.1fs',
          model_name, fasta_name, t_diff)

    plddt = prediction_result['plddt']
    ranking_confidences[model_name] = prediction_result['ranking_confidence']

    # Remove jax dependency from results.
    np_prediction_result = _jnp_to_np(dict(prediction_result))

    # Save the model outputs.
    result_output_path = os.path.join(current_prediction_dir, f'result_{model_name}.pkl')
    with open(result_output_path, 'wb') as f:
      pickle.dump(np_prediction_result, f, protocol=4)

    # Add the predicted LDDT in the b-factor column.
    # Note that higher predicted LDDT value means higher model confidence.
    plddt_b_factors = np.repeat(
        plddt[:, None], residue_constants.atom_type_num, axis=-1)
    unrelaxed_protein = protein.from_prediction(
        features=processed_feature_dict,
        result=prediction_result,
        b_factors=plddt_b_factors,
        remove_leading_feature_dimension=not model_runner.multimer_mode)

    unrelaxed_proteins[model_name] = unrelaxed_protein
    unrelaxed_pdbs[model_name] = protein.to_pdb(unrelaxed_protein)
    unrelaxed_pdb_path = os.path.join(current_prediction_dir, f'unrelaxed_{model_name}.pdb')
    with open(unrelaxed_pdb_path, 'w') as f:
      f.write(unrelaxed_pdbs[model_name])

  # Rank by model confidence.
  ranked_order = [
      model_name for model_name, confidence in
      sorted(ranking_confidences.items(), key=lambda x: x[1], reverse=True)]

  # Relax predictions.
  if models_to_relax == ModelsToRelax.BEST:
    to_relax = [ranked_order[0]]
  elif models_to_relax == ModelsToRelax.TOP5:
    to_relax = ranked_order[:5]
  elif models_to_relax == ModelsToRelax.ALL:
    to_relax = ranked_order
  elif models_to_relax == ModelsToRelax.NONE:
    to_relax = []


  try:
      for model_name in to_relax:
        t_0 = time.time()
        relaxed_pdb_str, _, violations = amber_relaxer.process(
            prot=unrelaxed_proteins[model_name])
        relax_metrics[model_name] = {
            'remaining_violations': violations,
            'remaining_violations_count': sum(violations)
        }
        timings[f'relax_{model_name}'] = time.time() - t_0

        relaxed_pdbs[model_name] = relaxed_pdb_str

        # Save the relaxed PDB.
        relaxed_output_path = os.path.join(current_prediction_dir, f'relaxed_{model_name}.pdb')
        with open(relaxed_output_path, 'w') as f:
          f.write(relaxed_pdb_str)
  except:
      pass

  # Write out relaxed PDBs in rank order.
  for idx, model_name in enumerate(ranked_order):
    ranked_output_path = os.path.join(current_prediction_dir, f'ranked_{idx}.pdb')
    with open(ranked_output_path, 'w') as f:
      if model_name in relaxed_pdbs:
        f.write(relaxed_pdbs[model_name])
      else:
        f.write(unrelaxed_pdbs[model_name])

  ranking_output_path = os.path.join(current_prediction_dir, 'ranking_debug.json')
  with open(ranking_output_path, 'w') as f:
    label = 'iptm+ptm' if 'iptm' in prediction_result else 'plddts'
    f.write(json.dumps(
        {label: ranking_confidences, 'order': ranked_order}, indent=4))

  logging.info('Final timings for %s: %s', fasta_name, timings)

  timings_output_path = os.path.join(current_prediction_dir, 'timings.json')
  with open(timings_output_path, 'w') as f:
    f.write(json.dumps(timings, indent=4))
  if models_to_relax != ModelsToRelax.NONE:
    relax_metrics_path = os.path.join(current_prediction_dir, 'relax_metrics.json')
    with open(relax_metrics_path, 'w') as f:
      f.write(json.dumps(relax_metrics, indent=4))


def main(argv):
  print("Hi Maryam!")

  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  for tool_name in (
      'jackhmmer', 'hhblits', 'hhsearch', 'hmmsearch', 'hmmbuild', 'kalign'):
    if not FLAGS[f'{tool_name}_binary_path'].value:
      raise ValueError(f'Could not find path to the "{tool_name}" binary. Make '
                       'sure it is installed on your system.')

  use_small_bfd = FLAGS.db_preset == 'reduced_dbs'
  _check_flag('small_bfd_database_path', 'db_preset',
              should_be_set=use_small_bfd)
  _check_flag('bfd_database_path', 'db_preset',
              should_be_set=not use_small_bfd)
  _check_flag('uniref30_database_path', 'db_preset',
              should_be_set=not use_small_bfd)

  run_multimer_system = 'multimer' in FLAGS.model_preset
# for multimers
  _check_flag('pdb_seqres_database_path', 'model_preset',
              should_be_set=run_multimer_system)
  _check_flag('uniprot_database_path', 'model_preset',
              should_be_set=run_multimer_system)

  run_monomer_system = 'monomer' in FLAGS.model_preset
# for monomers
  _check_flag('pdb70_database_path', 'model_preset',
              should_be_set=not run_multimer_system)

  _check_flag('precomputed_paired_msa_file', 'use_precomputed_paired_msa',
              should_be_set=FLAGS['use_precomputed_paired_msa'].value)


  if FLAGS.model_preset == 'monomer_casp14':
    num_ensemble = 8
  else:
    num_ensemble = 1

  if run_multimer_system:
    template_searcher = hmmsearch.Hmmsearch(
        binary_path=FLAGS.hmmsearch_binary_path,
        hmmbuild_binary_path=FLAGS.hmmbuild_binary_path,
        database_path=FLAGS.pdb_seqres_database_path)
    template_featurizer = templates.HmmsearchHitFeaturizer(
        mmcif_dir=FLAGS.template_mmcif_dir,
        max_template_date=FLAGS.max_template_date,
        max_hits=MAX_TEMPLATE_HITS,
        kalign_binary_path=FLAGS.kalign_binary_path,
        release_dates_path=None,
        obsolete_pdbs_path=FLAGS.obsolete_pdbs_path)
  else:
    template_searcher = hhsearch.HHSearch(
        binary_path=FLAGS.hhsearch_binary_path,
        databases=[FLAGS.pdb70_database_path])
    template_featurizer = templates.HhsearchHitFeaturizer(
        mmcif_dir=FLAGS.template_mmcif_dir,
        max_template_date=FLAGS.max_template_date,
        max_hits=MAX_TEMPLATE_HITS,
        kalign_binary_path=FLAGS.kalign_binary_path,
        release_dates_path=None,
        obsolete_pdbs_path=FLAGS.obsolete_pdbs_path)

  monomer_data_pipeline = pipeline.DataPipeline(
      jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
      hhblits_binary_path=FLAGS.hhblits_binary_path,
      uniref90_database_path=FLAGS.uniref90_database_path,
      mgnify_database_path=FLAGS.mgnify_database_path,
      bfd_database_path=FLAGS.bfd_database_path,
      uniref30_database_path=FLAGS.uniref30_database_path,
      small_bfd_database_path=FLAGS.small_bfd_database_path,
      template_searcher=template_searcher,
      template_featurizer=template_featurizer,
      use_small_bfd=use_small_bfd,
      use_precomputed_msas=FLAGS.use_precomputed_msas,
      use_precomputed_msas_from_dir = FLAGS.use_precomputed_msas_from_dir,
      use_templates = FLAGS.use_templates,
      no_MSA = FLAGS.no_MSA,
      save_chain_msa = FLAGS.save_chain_msa
  )

  if run_multimer_system:
    if FLAGS.use_precomputed_paired_msa:
          if (FLAGS.precomputed_paired_msa_file.split('.')[-1] != 'sto')\
                  and (FLAGS.precomputed_paired_msa_file.split('.')[-1] != 'txt'):
              raise ValueError(f'precomputed_paired_msa_file should have sto or txt file format')

    num_predictions_per_model = FLAGS.num_multimer_predictions_per_model
    data_pipeline = pipeline_multimer.DataPipeline(
        monomer_data_pipeline=monomer_data_pipeline,
        jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
        uniprot_database_path=FLAGS.uniprot_database_path,
        use_precomputed_msas=FLAGS.use_precomputed_msas,
        use_pairing= FLAGS.use_pairing,
        use_precomputed_paired_msa = FLAGS.use_precomputed_paired_msa,
        precomputed_paired_msa_file = FLAGS.precomputed_paired_msa_file,
        save_chain_msa= FLAGS.save_chain_msa,
        save_multimer_msa= FLAGS.save_multimer_msa
    )
  else:
    num_predictions_per_model = 1
    data_pipeline = monomer_data_pipeline

  model_runners = {}
  model_names = config.MODEL_PRESETS[FLAGS.model_preset]
  for model_name in model_names:
    model_config = config.model_config(model_name)
    if run_multimer_system:
      model_config.model.num_ensemble_eval = num_ensemble
    else:
      model_config.data.eval.num_ensemble = num_ensemble
    model_params = data.get_model_haiku_params(
        model_name=model_name, data_dir=FLAGS.data_dir)
    model_runner = model.RunModel(model_config, model_params)
    for i in range(num_predictions_per_model):
      model_runners[f'{model_name}_pred_{i}'] = model_runner

  logging.info('Have %d models: %s', len(model_runners),
               list(model_runners.keys()))

  amber_relaxer = relax.AmberRelaxation(
      max_iterations=RELAX_MAX_ITERATIONS,
      tolerance=RELAX_ENERGY_TOLERANCE,
      stiffness=RELAX_STIFFNESS,
      exclude_residues=RELAX_EXCLUDE_RESIDUES,
      max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS,
      use_gpu=FLAGS.use_gpu_relax)

  random_seed = FLAGS.random_seed
  if random_seed is None:
    random_seed = random.randrange(sys.maxsize // len(model_runners))
  logging.info('Using random seed %d for the data pipeline', random_seed)

  # Predict structure for each of the fasta sequences.
  is_fasta_path_dir = False
  #check to get contents of directory as fasta files if it is a dir name:
  if (FLAGS.fasta_paths)[0].endswith('.fasta'):
      fasta_names = [pathlib.Path(p).stem for p in FLAGS.fasta_paths]
  else:
      fasta_names = [s.strip('.fasta') for s in os.listdir(FLAGS.fasta_paths[0])]
      is_fasta_path_dir = True

  # Check for duplicate FASTA file names.
  if len(fasta_names) != len(set(fasta_names)):
    raise ValueError('All FASTA paths must have a unique basename.')

  for fasta_name in fasta_names:
    logging.info(f'***************************{fasta_name}********************************')
    if is_fasta_path_dir :
        path = FLAGS.fasta_paths[0]+'/'+fasta_name+'.fasta'
    else:
        path = FLAGS.fasta_paths[0]

    predict_structure(
        fasta_path=path,
        fasta_name=fasta_name,
        output_dir_base=FLAGS.output_dir,
        prediction_dir=FLAGS.prediction_dir,
        data_pipeline=data_pipeline,
        model_runners=model_runners,
        amber_relaxer=amber_relaxer,
        benchmark=FLAGS.benchmark,
        random_seed=random_seed,
        models_to_relax=FLAGS.models_to_relax)


    if is_fasta_path_dir:
        # if FLAGS.only_features:
            # move the fasta with MSAs generated to prediction folders
            # shutil.copy(path, '/projects/protein-structure-prediction/HostViralComplexes/Completed-Fastas')
            # shutil.copytree(f'/projects/protein-structure-prediction/HostViralComplexes/Predictions/NoPairing-Unrelaxed/{fasta_name}',
            #             f'/projects/protein-structure-prediction/HostViralComplexes/Predictions/Pairing-Unrelaxed/{fasta_name}')

            # shutil.copytree(f'/projects/protein-structure-prediction/HostViralComplexes/Predictions/NoPairing-Unrelaxed/{fasta_name}',
            #             f'/projects/protein-structure-prediction/HostViralComplexes/Predictions/Pairing-Unrelaxed-HVDB/{fasta_name}')
        #delete the predicted fasta file from fasta folder
        os.remove(path)

  logging.info("*************************END of CODE******************************")
if __name__ == '__main__':
  flags.mark_flags_as_required([
      'fasta_paths',
      'output_dir',
      'data_dir',
      'uniref90_database_path',
      'mgnify_database_path',
      'template_mmcif_dir',
      'max_template_date',
      'obsolete_pdbs_path',
      'use_gpu_relax',
  ])

  app.run(main)