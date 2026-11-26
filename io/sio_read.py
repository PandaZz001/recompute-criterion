# -*- coding: utf-8 -*-
"""SIO 声学数据格式读取器（sioread.m 的 Python 移植，官方格式）。

SIO 头（大端 int32）：ID, nr(记录数), rl(记录字节长), nc(通道数),
sl(字节/点: 4=float 2=short), f0(0=整型 1=实型), np(每通道点数),
bs(字节序常数 32677), fn(24 字符), com(72 字符) → 头共 132 字节。

数据记录（官方 sioread.m 口径）：记录 k 的文件偏移 = rl×k（k 从 1 起，
头 132 B 后填充到 rl 块对齐）；记录 i 的通道 j = 记录号 (j-1) + i·nc，
逐记录 rl 字节，每记录 ptrec = rl/sl 个点。

**差分编码（2026-11-25 发现，VLA/TLA 均适用）**：文件内存储的是相邻
采样差 d[n] = x[n] − x[n−1]（float32），原始波形需通道内累积和恢复
（cumsum 后块内 lag1 自相关 0.998，跨块差分链连续 0.996）。
本模块构建每通道块首累计表 offsets[ch, k]，随机读取时以块为单位
恢复：块首值 = offsets[ch, k] + 块内差分累积。
"""

import struct

import numpy as np


class SioFile:
    """SIO 读取器（差分编码自动恢复）。"""

    def __init__(self, path):
        with open(path, "rb") as f:
            self._data = f.read()
        self.endian = ">"
        if struct.unpack(">i", self._data[28:32])[0] != 32677:
            self.endian = "<"
            if struct.unpack("<i", self._data[28:32])[0] != 32677:
                raise ValueError("字节序常数 32677 校验失败：不是 SIO 文件？")
        self.e = self.endian
        hdr = struct.unpack(self.e + "9i", self._data[:36])
        self.id_, self.nr, self.rl, self.nc, self.sl = hdr[:5]
        self.f0, self.np, self.bs = hdr[5:8]
        self.filename = self._data[36:60].decode("ascii", "replace").strip()
        self.comment = self._data[60:132].decode("ascii", "replace").strip()
        self.ptrec = self.rl // self.sl          # 每记录点数
        self.rechan = -(-self.nr // self.nc)     # 每通道记录数（向上取整）
        self.dtype = np.float32 if self.sl == 4 else np.int16
        self.header_len = 132
        self._build_diff_offsets()

    def _build_diff_offsets(self):
        """构建每通道块首累计表（差分链前缀和）。

        offsets[ch, k] = 通道 ch（0 起）第 k 块首点前的全部差分和。
        记录 r（0 起）属于通道 r % nc、其块号 r // nc。
        """
        nrec = len(self._data) // self.rl - 1    # 数据区记录数
        nrec = min(nrec, self.nr)
        arr = np.frombuffer(self._data, dtype=self.dtype,
                            count=nrec * self.ptrec,
                            offset=self.rl)
        arr = arr.reshape(nrec, self.ptrec).astype(np.float64)
        self._offsets = np.zeros((self.nc, self.rechan + 1))
        for ch in range(self.nc):
            chain = arr[ch::self.nc].reshape(-1)   # 该通道全部差分
            nch = len(chain)
            block_sum = arr[ch::self.nc].sum(axis=1)
            self._offsets[ch, :len(block_sum) + 1] = \
                np.concatenate([[0.0], np.cumsum(block_sum)])
            if nch == 0:
                continue
            self._chain = None                      # 不驻留完整恢复波形
        # 记录数组引用释放（避免双份内存）
        del arr

    def read_channel(self, ch, p1=0, npi=None):
        """读通道 ch（1 起）的第 p1 点起 npi 个点（差分编码已恢复）。

        官方寻址：fseek(fid, rl*trec) —— 记录 k 的文件偏移 = rl×k
        （k 从 1 起；132 B 头后的填充区由 rl 块对齐覆盖，不参与寻址）。
        """
        if npi is None or npi <= 0:
            npi = self.np
        p2 = min(p1 + npi, self.np)
        out = np.empty(p2 - p1, dtype=np.float64)
        r1 = p1 // self.ptrec               # 0 起的块号
        pos = 0
        trec = r1 * self.nc + ch            # 1 起的记录号
        pp1 = p1 % self.ptrec
        acc = self._offsets[ch - 1, r1]     # 该块首点前的差分累计
        first = True
        while pos < p2 - p1:
            off = trec * self.rl            # 官方 fseek 口径
            rec = np.frombuffer(self._data[off:off + self.rl],
                                dtype=self.dtype).astype(np.float64)
            n = min(self.ptrec - pp1, p2 - p1 - pos)
            if first and pp1 > 0:
                acc += rec[:pp1].sum()      # 块内起点前的差分
                first = False
            seg = np.cumsum(rec[pp1:pp1 + n]) + acc
            out[pos:pos + n] = seg
            pos += n
            acc = seg[-1] if n else acc
            pp1 = 0
            trec += self.nc
        return out
