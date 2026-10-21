// Marching-squares contour extraction for drawing level curves as plotly
// Scatter/Scatter3d lines (plotly's own contour trace can't live in a 3D scene).
//
//   contourLines(xs, ys, z, level, { join = true })  -> { x, y }
//     xs: length nx, ys: length ny, z[j][i] = f(xs[i], ys[j]) (plotly convention).
//     Returns flat arrays with null between polylines. Closed loops repeat their
//     first point at the end. Cells with any non-finite corner are skipped.
//   contourLevels(z, n) -> n evenly spaced levels strictly inside (min, max)
//     of the finite values: min + k (max - min) / (n + 1), k = 1..n.

export function contourLines(xs, ys, z, level, { join = true } = {}) {
  const nx = xs.length;
  const ny = ys.length;
  const outX = [];
  const outY = [];
  if (nx < 2 || ny < 2 || !Number.isFinite(level)) return { x: outX, y: outY };

  // Flatten once: value grid, plus above/finite flags per node.
  const n = nx * ny;
  const zz = new Float64Array(n);
  const up = new Uint8Array(n);
  const ok = new Uint8Array(n);
  for (let j = 0; j < ny; j++) {
    const row = z[j];
    for (let i = 0; i < nx; i++) {
      const k = j * nx + i;
      const v = row[i];
      zz[k] = v;
      if (Number.isFinite(v)) {
        ok[k] = 1;
        up[k] = v >= level ? 1 : 0;
      }
    }
  }

  // Edge ids: horizontal edge (i,j)-(i+1,j) -> j*(nx-1)+i;
  // vertical edge (i,j)-(i,j+1) -> H + j*nx + i.
  const H = (nx - 1) * ny;
  const nEdges = H + nx * (ny - 1);
  const nb0 = new Int32Array(nEdges).fill(-1);
  const nb1 = new Int32Array(nEdges).fill(-1);
  const segs = join ? null : [];

  const link = (a, b) => {
    if (segs) {
      segs.push(a, b);
      return;
    }
    if (nb0[a] < 0) nb0[a] = b;
    else nb1[a] = b;
    if (nb0[b] < 0) nb0[b] = a;
    else nb1[b] = a;
  };

  for (let j = 0; j < ny - 1; j++) {
    for (let i = 0; i < nx - 1; i++) {
      const k0 = j * nx + i; // bottom-left
      const k1 = k0 + 1; // bottom-right
      const k3 = k0 + nx; // top-left
      const k2 = k3 + 1; // top-right
      if (!(ok[k0] & ok[k1] & ok[k2] & ok[k3])) continue;
      const c = up[k0] | (up[k1] << 1) | (up[k2] << 2) | (up[k3] << 3);
      if (c === 0 || c === 15) continue;
      const eB = j * (nx - 1) + i; // bottom: k0-k1
      const eT = eB + (nx - 1); // top: k3-k2
      const eL = H + j * nx + i; // left: k0-k3
      const eR = eL + 1; // right: k1-k2
      switch (c) {
        case 1: case 14: link(eL, eB); break;
        case 2: case 13: link(eB, eR); break;
        case 3: case 12: link(eL, eR); break;
        case 4: case 11: link(eR, eT); break;
        case 6: case 9: link(eB, eT); break;
        case 7: case 8: link(eL, eT); break;
        case 5: case 10: {
          // Saddle: the cell-center average decides which diagonal pair is joined.
          const centerUp = (zz[k0] + zz[k1] + zz[k2] + zz[k3]) / 4 >= level;
          // c=5: k0,k2 up. If center is up they're connected, so cut off k1 and k3.
          if ((c === 5) === centerUp) {
            link(eB, eR); // around k1
            link(eT, eL); // around k3
          } else {
            link(eL, eB); // around k0
            link(eR, eT); // around k2
          }
          break;
        }
      }
    }
  }

  const emit = (e) => {
    let a, b, x, y;
    if (e < H) {
      const j = Math.floor(e / (nx - 1));
      const i = e - j * (nx - 1);
      a = j * nx + i;
      b = a + 1;
      const t = (level - zz[a]) / (zz[b] - zz[a]);
      x = xs[i] + t * (xs[i + 1] - xs[i]);
      y = ys[j];
    } else {
      const v = e - H;
      const j = Math.floor(v / nx);
      const i = v - j * nx;
      a = v;
      b = a + nx;
      const t = (level - zz[a]) / (zz[b] - zz[a]);
      x = xs[i];
      y = ys[j] + t * (ys[j + 1] - ys[j]);
    }
    outX.push(x);
    outY.push(y);
  };
  const sep = () => {
    if (outX.length) {
      outX.push(null);
      outY.push(null);
    }
  };

  if (segs) {
    for (let s = 0; s < segs.length; s += 2) {
      sep();
      emit(segs[s]);
      emit(segs[s + 1]);
    }
    return { x: outX, y: outY };
  }

  const seen = new Uint8Array(nEdges);
  const walk = (start) => {
    sep();
    let prev = -1;
    let cur = start;
    for (;;) {
      seen[cur] = 1;
      emit(cur);
      const next = nb0[cur] !== prev ? nb0[cur] : nb1[cur];
      if (next < 0) return;
      if (seen[next]) {
        if (next === start) emit(start); // close the loop
        return;
      }
      prev = cur;
      cur = next;
    }
  };
  // Open curves first (endpoints have one neighbour), then the remaining loops.
  for (let e = 0; e < nEdges; e++) if (nb0[e] >= 0 && nb1[e] < 0 && !seen[e]) walk(e);
  for (let e = 0; e < nEdges; e++) if (nb1[e] >= 0 && !seen[e]) walk(e);
  return { x: outX, y: outY };
}

export function contourLevels(z, n) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const row of z) {
    for (const v of row) {
      if (Number.isFinite(v)) {
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
  }
  if (!(hi > lo) || !(n >= 1)) return [];
  const step = (hi - lo) / (n + 1);
  return Array.from({ length: n }, (_, k) => lo + (k + 1) * step);
}
