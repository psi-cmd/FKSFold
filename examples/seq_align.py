"""
standalone_seqalign.py

Standalone sequence alignment and index‐mapping utilities,
based on PyMOL 的 seqalign 模块实现，依赖 BioPython。

Usage:
    from standalone_seqalign import map_indices

    seq1 = "ACDEFGHIKLMNPQRS"
    seq2 = "ACDIKLMNP"
    mapping = map_indices(seq1, seq2)
    # mapping 是一个 [(i_in_seq1, j_in_seq2), ...] 列表
"""

import functools
from Bio.Align import PairwiseAligner, MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import substitution_matrices

@functools.lru_cache(maxsize=None)
def _get_aligner_blosum62():
    """返回一个 Needleman–Wunsch 全局比对器，使用 BLOSUM62 矩阵。"""
    blosum62 = substitution_matrices.load("BLOSUM62")
    # 增加特殊字符到矩阵字母表
    missing = ''.join(set('JUO-.?') - set(blosum62.alphabet))
    blosum62 = blosum62.select(blosum62.alphabet + missing)
    aligner = PairwiseAligner(
        internal_open_gap_score=-10,
        extend_gap_score=-0.5,
        substitution_matrix=blosum62
    )
    aligner.mode = "global"
    return aligner

def needle_alignment(s1: str, s2: str) -> MultipleSeqAlignment:
    """
    对两个序列做 Needleman–Wunsch 全局比对，
    返回 Bio.Align.MultipleSeqAlignment 对象（两条序列的比对结果）。
    """
    aligner = _get_aligner_blosum62()
    alns = aligner.align(s1, s2)
    # 取第一个最优比对
    aln1, aln2 = alns[0]
    rec1 = SeqRecord(Seq(aln1), id="seq1")
    rec2 = SeqRecord(Seq(aln2), id="seq2")
    msa = MultipleSeqAlignment([rec1, rec2])
    return msa

def alignment_mapping(seq1_aln: str, seq2_aln: str):
    """
    接受两条带 '-' 的对齐后序列，将非 gap 位点 i->j 的映射以迭代方式返回。
    """
    i = j = -1
    for a, b in zip(seq1_aln, seq2_aln):
        if a != '-':
            i += 1
        if b != '-':
            j += 1
        if a != '-' and b != '-':
            yield i, j

def map_indices(seq1: str, seq2: str):
    """
    对两条原始序列做比对并返回索引映射列表。
    返回值示例：
        [(0,0), (1,1), (2,2), (5,3), ...]
    表示 seq1[0] 对应 seq2[0]，seq1[1] 对应 seq2[1]，以此类推。
    """
    msa = needle_alignment(seq1, seq2)
    s1_aln = str(msa[0].seq)
    s2_aln = str(msa[1].seq)
    return list(alignment_mapping(s1_aln, s2_aln))