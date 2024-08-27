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

"""Functions for building the input features for the AlphaFold model."""

import os
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Union
from absl import logging
from alphafold.common import residue_constants
from alphafold.data import msa_identifiers
from alphafold.data import parsers
from alphafold.data import templates
from alphafold.data.parsers import Msa
from alphafold.data.tools import hhblits
from alphafold.data.tools import hhsearch
from alphafold.data.tools import hmmsearch
from alphafold.data.tools import jackhmmer
import numpy as np
import pandas as pd
import json

# Internal import (7716).

FeatureDict = MutableMapping[str, np.ndarray]
TemplateSearcher = Union[hhsearch.HHSearch, hmmsearch.Hmmsearch]


def make_sequence_features(
    sequence: str, description: str, num_res: int) -> FeatureDict:
  """Constructs a feature dict of sequence features."""
  features = {}
  features['aatype'] = residue_constants.sequence_to_onehot(
      sequence=sequence,
      mapping=residue_constants.restype_order_with_x,
      map_unknown_to_x=True)
  features['between_segment_residues'] = np.zeros((num_res,), dtype=np.int32)
  features['domain_name'] = np.array([description.encode('utf-8')],
                                     dtype=np.object_)
  features['residue_index'] = np.array(range(num_res), dtype=np.int32)
  features['seq_length'] = np.array([num_res] * num_res, dtype=np.int32)
  features['sequence'] = np.array([sequence.encode('utf-8')], dtype=np.object_)
  return features

def make_msa_features(msas: Sequence[parsers.Msa], msa_output_dir, file_name, save_chain_msa) -> FeatureDict:
  """Constructs a feature dict of MSA features."""
  if not msas:
    raise ValueError('At least one MSA must be provided.')

  int_msa = []
  seq_msa=[]
  seq_original_msa = []
  deletion_matrix = []
  species_ids = []
  seen_sequences = set()
  msa_props = []

  for msa_index, msa in enumerate(msas):
    if not msa:
      raise ValueError(f'MSA {msa_index} must contain at least one sequence.')
    msa_prop = {}
    msa_prop['msa_index'] = msa_index
    msa_prop['total_seq_count'] = len(msa.sequences)
    msa_new_seq_count = 0
    for sequence_index, sequence in enumerate(msa.sequences):
      if sequence in seen_sequences:
        continue
      seen_sequences.add(sequence)
      msa_new_seq_count +=1
      int_msa.append(
          [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence])
      seq_msa.append(sequence)
      deletion_matrix.append(msa.deletion_matrix[sequence_index])
      identifiers = msa_identifiers.get_identifiers(
          msa.descriptions[sequence_index])
      species_ids.append(identifiers.species_id.encode('utf-8'))

      if len(msa.original_sequences) > 0:
        seq_original_msa.append(msa.original_sequences[sequence_index])
    msa_prop['msa_new_seq_count'] = msa_new_seq_count
    msa_props.append(msa_prop)
  num_res = len(msas[0].sequences[0])
  num_alignments = len(int_msa)
  features = {'deletion_matrix_int': np.array(deletion_matrix, dtype=np.int32),
              'msa': np.array(int_msa, dtype=np.int32),
              'seq_original_msa': np.array(seq_original_msa, dtype=np.object_),
              'seq_msa': np.array(seq_msa, dtype=np.object_),
              'num_alignments': np.array([num_alignments] * num_res, dtype=np.int32),
              'msa_species_identifiers': np.array(species_ids, dtype=np.object_)}

  supplementary_path = os.path.join(msa_output_dir, 'supplementary_files')
  os.makedirs(supplementary_path, exist_ok=True)
  save_msa_properties(msa_props, supplementary_path, file_name)

  if save_chain_msa:
      save_chain_msa_to_file(features, msa_output_dir, file_name)
  return features


def run_msa_tool(msa_runner, input_fasta_path: str, msa_out_path: str,
                 msa_format: str, use_precomputed_msas: bool,
                 max_sto_sequences: Optional[int] = None
                 ) -> Mapping[str, Any]:
  """Runs an MSA tool, checking if output already exists first."""
  if not use_precomputed_msas or not os.path.exists(msa_out_path):
    if msa_format == 'sto' and max_sto_sequences is not None:
      result = msa_runner.query(input_fasta_path, max_sto_sequences)[0]  # pytype: disable=wrong-arg-count
    else:
      result = msa_runner.query(input_fasta_path)[0]
    with open(msa_out_path, 'w') as f:
      f.write(result[msa_format])
  else:
    result = read_msa(msa_format, msa_out_path, max_sto_sequences)
  return result


def read_msa(msa_format, msa_out_path, max_sto_sequences: Optional[int] = None):
    logging.warning('Reading MSA from file %s', msa_out_path)
    if msa_format == 'sto' and max_sto_sequences is not None:
        precomputed_msa = parsers.truncate_stockholm_msa(
            msa_out_path, max_sto_sequences)
        result = {'sto': precomputed_msa}
    else:
        with open(msa_out_path, 'r') as f:
            result = {msa_format: f.read()}
    return result

def read_fasta_file(fasta_file_path: str):
  with open(fasta_file_path, "r") as fasta_file:
      lines = fasta_file.readlines()
      for line in lines:
          if line.startswith(">"):  # Header line, skipping it
              header = line[1:].strip()
          else:
              sequence = line.strip()
  return header, sequence

# save properties of each MSA database:
def save_msa_properties(msa_props, supplementary_path, file_name):
  # save msa properties to file
  with open(os.path.join(supplementary_path, f"{file_name}_props.txt"), 'w') as outfile:
      json.dump(msa_props, outfile)

def save_chain_msa_to_file(msa_features, msa_dir, file_name):
  msa_df_to_save = pd.DataFrame({
      'msa': [" ".join([str(res) for res in seq]) for seq in msa_features['msa']],
      'seq_msa': msa_features['seq_msa'],
      'msa_species_identifiers': msa_features['msa_species_identifiers']
  })
  msa_df_to_save.to_csv(os.path.join(msa_dir, f"final_{file_name}.csv"), index=False)

  file_msa = os.path.join(msa_dir, f"final_{file_name}.aln")
  with open(file_msa, 'w') as file:
      for item in msa_features['seq_msa']:
          file.write(str(item) + '\n')


class DataPipeline:
  """Runs the alignment tools and assembles the input features."""

  def __init__(self,
               jackhmmer_binary_path: str,
               hhblits_binary_path: str,
               uniref90_database_path: str,
               mgnify_database_path: str,
               bfd_database_path: Optional[str],
               uniref30_database_path: Optional[str],
               small_bfd_database_path: Optional[str],
               template_searcher: TemplateSearcher,
               template_featurizer: templates.TemplateHitFeaturizer,
               use_small_bfd: bool,
               mgnify_max_hits: int = 501,
               uniref_max_hits: int = 10000,
               use_precomputed_msas: bool = False,
               use_precomputed_msas_from_dir: bool = False,
               use_templates: bool = True,
               no_MSA: bool = False,
               save_chain_msa:bool = False):
    """Initializes the data pipeline."""
    self._use_small_bfd = use_small_bfd
    self.jackhmmer_uniref90_runner = jackhmmer.Jackhmmer(
        binary_path=jackhmmer_binary_path,
        database_path=uniref90_database_path)
    if use_small_bfd:
      self.jackhmmer_small_bfd_runner = jackhmmer.Jackhmmer(
          binary_path=jackhmmer_binary_path,
          database_path=small_bfd_database_path)
    else:
      self.hhblits_bfd_uniref_runner = hhblits.HHBlits(
          binary_path=hhblits_binary_path,
          databases=[bfd_database_path, uniref30_database_path])
    self.jackhmmer_mgnify_runner = jackhmmer.Jackhmmer(
        binary_path=jackhmmer_binary_path,
        database_path=mgnify_database_path)
    self.template_searcher = template_searcher
    self.template_featurizer = template_featurizer
    self.mgnify_max_hits = mgnify_max_hits
    self.uniref_max_hits = uniref_max_hits
    self.use_precomputed_msas = use_precomputed_msas
    self.use_precomputed_msas_from_dir = use_precomputed_msas_from_dir
    self.use_templates = use_templates
    self.no_MSA = no_MSA
    self.save_chain_msa = save_chain_msa

  def process(self, input_fasta_path: str, msa_output_dir: str) -> FeatureDict:
    """Runs alignment tools on the input sequence and creates features."""
    with open(input_fasta_path) as f:
      input_fasta_str = f.read()
    input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
    if len(input_seqs) != 1:
      raise ValueError(
          f'More than one input sequence found in {input_fasta_path}.')
    input_sequence = input_seqs[0]

    sequence_features = make_sequence_features(
        sequence=input_sequence,
        description=input_descs[0],
        num_res=len(input_sequence))
    uniref90_result_sto = None
    if self.no_MSA:
        header, sequence = read_fasta_file(input_fasta_path)
        length = len(sequence)
        msa = [Msa(sequences=[sequence],
            deletion_matrix=[np.zeros(length, dtype=int).tolist()],
            descriptions=[header])]
    else:
        if (not self.use_precomputed_msas_from_dir) or self.use_templates:
            uniref90_result_sto = self._get_uniref90_MSA(input_fasta_path, msa_output_dir)

        if not self.use_precomputed_msas_from_dir:
            #find MSA features based on uniref90, mgnify and bfd
            uniref90_msa = parsers.parse_stockholm(uniref90_result_sto)
            logging.info('Uniref90 MSA size: %d sequences.', len(uniref90_msa))

            mgnify_msa = self._get_mgnigy_MSA(input_fasta_path, msa_output_dir)
            logging.info('MGnify MSA size: %d sequences.', len(mgnify_msa))

            bfd_msa = self._get_bfd_MSA(input_fasta_path, msa_output_dir)
            logging.info('BFD MSA size: %d sequences.', len(bfd_msa))
            msa = (uniref90_msa, bfd_msa, mgnify_msa)
        else: #use_precomputed_msas_from_dir
            msa = []
            # find MSA features based on pre_computed MSAs
            precomputed_msa_path = os.path.join(msa_output_dir, 'MSAs')
            for msa_file in sorted(os.listdir(precomputed_msa_path)):
                msa_format = msa_file.split('.')[-1]
                msa_read = read_msa(msa_format, os.path.join(precomputed_msa_path, msa_file))
                if msa_format == 'sto':
                    parsed_msa = parsers.parse_stockholm(msa_read[msa_format])
                elif msa_format == 'a3m':
                    parsed_msa = parsers.parse_a3m(msa_read[msa_format])
                else:
                    continue
                msa.append(parsed_msa)

    msa_features = make_msa_features(msa, msa_output_dir, "individual_msa", self.save_chain_msa)

    # Find template features based on uniref90 MSAs
    templates_result = self._find_templates(input_sequence, msa_output_dir, uniref90_result_sto)
    logging.info('Total number of templates (NB: this can include bad '
                 'templates and is later filtered to top 4): %d.',
                 templates_result.features['template_domain_names'].shape[0])

    logging.info('Final (deduplicated) MSA size: %d sequences.',
                 msa_features['num_alignments'][0])

    return {**sequence_features, **msa_features, **templates_result.features}

  def _get_bfd_MSA(self, input_fasta_path, msa_output_dir):
      if self._use_small_bfd:
          bfd_out_path = os.path.join(msa_output_dir, 'small_bfd_hits.sto')
          jackhmmer_small_bfd_result = run_msa_tool(
              msa_runner=self.jackhmmer_small_bfd_runner,
              input_fasta_path=input_fasta_path,
              msa_out_path=bfd_out_path,
              msa_format='sto',
              use_precomputed_msas=self.use_precomputed_msas)
          bfd_msa = parsers.parse_stockholm(jackhmmer_small_bfd_result['sto'])
      else:
          bfd_out_path = os.path.join(msa_output_dir, 'bfd_uniclust_hits.a3m')
          hhblits_bfd_uniref_result = run_msa_tool(
              msa_runner=self.hhblits_bfd_uniref_runner,
              input_fasta_path=input_fasta_path,
              msa_out_path=bfd_out_path,
              msa_format='a3m',
              use_precomputed_msas=self.use_precomputed_msas)
          bfd_msa = parsers.parse_a3m(hhblits_bfd_uniref_result['a3m'])
      return bfd_msa

  def _find_templates(self, input_sequence, msa_output_dir, msa_for_templates):
      if self.use_templates:
          msa_for_templates = parsers.deduplicate_stockholm_msa(msa_for_templates)
          msa_for_templates = parsers.remove_empty_columns_from_stockholm_msa(
              msa_for_templates)
          if self.template_searcher.input_format == 'sto':
              pdb_templates_result = self.template_searcher.query(msa_for_templates)
          elif self.template_searcher.input_format == 'a3m':
              uniref90_msa_as_a3m = parsers.convert_stockholm_to_a3m(msa_for_templates)
              pdb_templates_result = self.template_searcher.query(uniref90_msa_as_a3m)
          else:
              raise ValueError('Unrecognized template input format: '
                               f'{self.template_searcher.input_format}')
          pdb_hits_out_path = os.path.join(
              msa_output_dir, f'pdb_hits.{self.template_searcher.output_format}')
          with open(pdb_hits_out_path, 'w') as f:
              f.write(pdb_templates_result)

          pdb_template_hits = self.template_searcher.get_template_hits(
              output_string=pdb_templates_result, input_sequence=input_sequence)
          templates_result = self.template_featurizer.get_templates(
              query_sequence=input_sequence,
              hits=pdb_template_hits)
      else:
          templates_result = self.template_featurizer.get_templates(
              query_sequence=input_sequence,
              hits=[])
      return templates_result

  def _get_mgnigy_MSA(self, input_fasta_path, msa_output_dir):
      mgnify_out_path = os.path.join(msa_output_dir, 'mgnify_hits.sto')
      jackhmmer_mgnify_result = run_msa_tool(
          msa_runner=self.jackhmmer_mgnify_runner,
          input_fasta_path=input_fasta_path,
          msa_out_path=mgnify_out_path,
          msa_format='sto',
          use_precomputed_msas=self.use_precomputed_msas,
          max_sto_sequences=self.mgnify_max_hits)
      return parsers.parse_stockholm(jackhmmer_mgnify_result['sto'])

  def _get_uniref90_MSA(self, input_fasta_path, msa_output_dir):
      uniref90_out_path = os.path.join(msa_output_dir, 'uniref90_hits.sto')
      jackhmmer_uniref90_result = run_msa_tool(
          msa_runner=self.jackhmmer_uniref90_runner,
          input_fasta_path=input_fasta_path,
          msa_out_path=uniref90_out_path,
          msa_format='sto',
          use_precomputed_msas=self.use_precomputed_msas,
          max_sto_sequences=self.uniref_max_hits)
      return jackhmmer_uniref90_result['sto']
