import { describe, expect, it } from "vitest";
import { CALCULUS_SURFACES, getSurface, surfaceGrid } from "../src/lib/surfaces.js";
import { contourLines } from "../src/lib/contour.js";
import { XY_PAD, Z_PAD, clip, fieldArrows, fieldData, levelAt, npGradient, pointGeometry } from "../src/demos/gradient_field.js";

// Reference values from the ORIGINAL Python (GradientFieldVisualization in
// content/Chapter_09/utils_lsg.py), run with ~/miniforge3/bin/python: per surface,
// zmin/zmax from _update_z_stats; gx/gy = np.gradient(Z, y, x) at [0,0], [37,91],
// [159,159]; the traces of _add_gradient_field(fig, density=14) (nShaft = length of
// the shaft x array incl. NaN separators, z = floor height, *X/*Y = nansum of
// shaft / head coordinates). For two points, partial_derivatives(), and the traces
// of _add_normal_line(length=1.6), _add_tangent_traces(half_len=0.9) (first/last x
// or y and z), _add_gradient_vector(length=0.6) (lifted and floor end points), and
// the level set at clip(f(x0, y0), zmin, zmax): contourpy polyline count and total length.
const REF = {
  "Original (sin/cos + saddle)": {"zmin":-1.6172456389636483,"zmax":1.4204940449856505,"nShaft":675,"z":-1.6162456389636484,"shaftX":-45.351393057868194,"shaftY":-41.28956548156998,"headX":-95.04099366630969,"headY":-80.83810136901263,"gx":[-0.4057311869089355,0.11523469393331442,1.3829480583740843],"gy":[0.8830667161173772,0.6912024475972922,-0.9056125291656433],"pts":{"0.5,0.5":{"z0":0.21036774620197413,"fx":0.5350755122877776,"fy":-0.2649244043788912,"normal":[[0.13246674097665667,0.8675332590233433],[0.6819715675641397,0.3180284324358603],[0.8972488509864411,-0.47651335858249283]],"tanX":[-0.4,1.4,-0.2712002148570257,0.691935707260974],"tanY":[-0.4,1.4,0.4487997101429762,-0.028064217739027925],"lift":[0.961672259587587,0.2714187145912263,0.5179540278989387],"floor":[1.0377026753210257,0.23377484537968285,-1.6162456389636484],"level":0.21036774620197413,"nPaths":2,"len":14.420970850921076},"-1.3,2.1":{"z0":-0.1647757017684114,"fx":-0.45752285955420746,"fy":-0.21412384383102945,"normal":[[-0.9732990185290229,-1.6267009814709772],[2.2528983054619314,1.9471016945380688],[0.5492891250454923,-0.8788405285823151]],"tanX":[-2.2,-0.4,0.24699487183037538,-0.5765462753671982],"tanY":[1.2000000000000002,3.0,0.02793575767951509,-0.35748716121633783],"lift":[-1.7850557199539634,1.8729905882080535,0.10575650610676335],"floor":[-1.8434304581204377,1.8456708094083356,-1.6162456389636484],"level":-0.1647757017684114,"nPaths":2,"len":16.709137026242125}}},
  "Monkey saddle": {"zmin":-3.2399999999999998,"zmax":3.2399999999999998,"nShaft":675,"z":-3.239,"shaftX":-42.462042663871685,"shaftY":-42.61745543785139,"headX":-84.93787341212952,"headY":-85.48130142550394,"gx":[-0.020291918832301196,-0.42899252402990395,-0.020291918832312725],"gy":[-3.219622641509429,0.25055179779280845,-3.2196226415094147],"pts":{"0.5,0.5":{"z0":-0.015,"fx":5.999999802552836e-08,"fy":-0.08999999999999807,"normal":[[0.4999999521932285,0.5000000478067715],[0.5717101595967108,0.4282898404032892],[0.7817795510745817,-0.8117795510745818]],"tanX":[-0.4,1.4,-0.015000053999998222,-0.014999946000001776],"tanY":[-0.4,1.4,0.06599999999999827,-0.09599999999999825],"lift":[0.5000003983897624,-0.09758466330580351,0.03878261969754506],"floor":[0.5000003999999868,-0.09999999999986664,-3.239],"level":-0.015,"nPaths":3,"len":17.034112344156092},"-1.3,2.1":{"z0":0.9001200000000001,"fx":-0.489599939999974,"fy":0.9827999999998394,"normal":[[-1.0362651154175369,-1.5637348845824632],[1.5705909388640524,2.6294090611359477],[1.4387942583801734,0.361445741619827]],"tanX":[-2.2,-0.4,1.3407599459999768,0.45948005400002356],"tanY":[1.2000000000000002,3.0,0.015600000000144831,1.7846399999998555],"lift":[-1.4801467834759017,2.4616182199697505,1.3437182409672022],"floor":[-1.5675409573386547,2.6370491934137097,-3.239],"level":0.9001200000000001,"nPaths":3,"len":8.47788910476401}}},
  "Paraboloid": {"zmin":8.543965824136499e-05,"zmax":2.16,"nShaft":675,"z":0.001085439658241365,"shaftX":-43.71126285171652,"shaftY":-43.711262851716526,"headX":-89.305990829792,"headY":-89.30599082979211,"gx":[-0.7154716981132155,0.10415094339622577,0.7154716981132072],"gy":[-0.7154716981132155,-0.38490566037735885,0.7154716981132072],"pts":{"0.5,0.5":{"z0":0.06,"fx":0.11999999999999858,"fy":0.11999999999999858,"normal":[[0.4053532391929892,0.5946467608070107],[0.4053532391929892,0.5946467608070107],[0.8487230067250993,-0.7287230067250992]],"tanX":[-0.4,1.4,-0.047999999999998724,0.1679999999999987],"tanY":[-0.4,1.4,-0.047999999999998724,0.1679999999999987],"lift":[0.9182835398998704,0.9182835398998704,0.16038804957596775],"floor":[0.9242640687119286,0.9242640687119286,0.001085439658241365],"level":0.06,"nPaths":1,"len":4.441340448617994},"-1.3,2.1":{"z0":0.732,"fx":-0.31199999999997896,"fy":0.5039999999999489,"normal":[[-1.085286648416251,-1.514713351583749],[1.753155355133956,2.446844644866044],[1.4201838191787288,0.043816180821271145]],"tanX":[-2.2,-0.4,1.012799999999981,0.4512000000000189],"tanY":[1.2000000000000002,3.0,0.278400000000046,1.185599999999954],"lift":[-1.5716715466967492,2.538854036971657,1.0379439572030726],"floor":[-1.615812768769785,2.610159088012712,0.001085439658241365],"level":0.732,"nPaths":1,"len":15.51786892260435}}},
  "Sine product": {"zmin":-1.0,"zmax":1.0,"nShaft":675,"z":-0.999,"shaftX":-42.96915091699941,"shaftY":-42.96915091699947,"headX":-86.71106633254583,"headY":-86.71106633254578,"gx":[-0.04654110824548125,-0.7106791937710728,0.04654110824548365],"gy":[-0.04654110824548125,-0.8036796715919738,0.04654110824548365],"pts":{"0.5,0.5":{"z0":0.4999999999999999,"fx":0.7853978404154249,"fy":0.7853978404154249,"normal":[[0.07959554185836432,0.9204044581416357],[0.07959554185836432,0.9204044581416357],[1.0352758010122216,-0.03527580101222183]],"tanX":[-0.4,1.4,-0.20685805637388255,1.2068580563738822],"tanY":[-0.4,1.4,-0.20685805637388255,1.2068580563738822],"lift":[0.7838728615256022,0.7838728615256022,0.9459062647895098],"floor":[0.9242640687119286,0.9242640687119286,-0.999],"level":0.4999999999999999,"nPaths":8,"len":19.452028391049836},"-1.3,2.1":{"z0":0.13938412895876273,"fx":0.11155753376976274,"fy":1.3823579342526482,"normal":[[-1.3521972772734492,-1.247802722726551],[1.4532008287826161,2.746799171217384],[0.6072797117095983,-0.32851145379207275]],"tanX":[-2.2,-0.4,0.03898234856597625,0.2397859093515492],"tanY":[1.2000000000000002,3.0,-1.1047380118686205,1.383506269786146],"lift":[-1.2717720728498108,2.4497845259299424,0.6260605815931275],"floor":[-1.2517363647944995,2.698055701015173,-0.999],"level":0.13938412895876273,"nPaths":8,"len":28.37185036311627}}}
};

const sumFinite = (a) => a.reduce((t, v) => (v === null ? t : t + v), 0);
function stats({ x, y }) {
  let len = 0;
  let n = x.length ? 1 : 0;
  for (let i = 1; i < x.length; i++) {
    if (x[i] === null) n++;
    else if (x[i - 1] !== null) len += Math.hypot(x[i] - x[i - 1], y[i] - y[i - 1]);
  }
  return { len, n };
}
const close = (a, b, digits = 9) => a.forEach((v, i) => expect(v).toBeCloseTo(b[i], digits));

describe("gradient_field", () => {
  it("offers the 4 calculus surfaces in Python's order", () => {
    expect(CALCULUS_SURFACES.map((s) => s.name)).toEqual(Object.keys(REF));
  });

  it("clamps the point to the grid", () => {
    expect(clip(7, -3, 3)).toBe(3);
    expect(clip(-9, -3, 3)).toBe(-3);
    expect(clip(0.5, -3, 3)).toBe(0.5);
  });

  for (const s of CALCULUS_SURFACES) {
    const r = REF[s.name];
    it(`matches np.gradient and the floor arrows: ${s.name}`, () => {
      const d = fieldData(s);
      expect(d.zmin).toBeCloseTo(r.zmin, 10);
      expect(d.zmax).toBeCloseTo(r.zmax, 10);
      expect(d.zFloor).toBeCloseTo(r.z, 10);
      const { gx, gy } = npGradient(d.z, d.axis);
      close([gx[0][0], gx[37][91], gx[159][159]], r.gx);
      close([gy[0][0], gy[37][91], gy[159][159]], r.gy);
      expect(d.arrows.shafts.x.length).toBe(r.nShaft); // 15 x 15 arrows, 3 entries each
      expect(d.arrows.heads.x.length).toBe(2 * r.nShaft);
      expect(sumFinite(d.arrows.shafts.x)).toBeCloseTo(r.shaftX, 6);
      expect(sumFinite(d.arrows.shafts.y)).toBeCloseTo(r.shaftY, 6);
      expect(sumFinite(d.arrows.heads.x)).toBeCloseTo(r.headX, 6);
      expect(sumFinite(d.arrows.heads.y)).toBeCloseTo(r.headY, 6);
      expect(d.shaftZ.every((v) => v === null || v === d.zFloor)).toBe(true);
    });

    for (const [key, q] of Object.entries(r.pts)) {
      const [x0, y0] = key.split(",").map(Number);
      it(`matches the point geometry at (${key}): ${s.name}`, () => {
        const d = fieldData(s);
        const p = pointGeometry(s.f, x0, y0, d.zFloor);
        close([p.z0, p.fx, p.fy], [q.z0, q.fx, q.fy]);
        close(p.normal.x, q.normal[0]);
        close(p.normal.y, q.normal[1]);
        close(p.normal.z, q.normal[2]);
        close([p.tanX.x[0], p.tanX.x[59], p.tanX.z[0], p.tanX.z[59]], q.tanX);
        close([p.tanY.y[0], p.tanY.y[59], p.tanY.z[0], p.tanY.z[59]], q.tanY);
        close([p.lifted.x[1], p.lifted.y[1], p.lifted.z[1]], q.lift);
        close([p.floor.x[1], p.floor.y[1], p.floor.z[1]], q.floor);
        // Tangent plane passes through the point with the tangent lines' slopes.
        const P = p.plane;
        expect(P.z[0][1] - P.z[0][0]).toBeCloseTo(p.fx * (P.x[1] - P.x[0]), 9);
        expect(P.z[1][0] - P.z[0][0]).toBeCloseTo(p.fy * (P.y[1] - P.y[0]), 9);
        const level = levelAt(s.f, x0, y0, d.zmin, d.zmax);
        expect(level).toBeCloseTo(q.level, 12);
        const st = stats(contourLines(d.axis, d.axis, d.z, level));
        expect(st.n).toBe(q.nPaths);
        expect(Math.abs(st.len - q.len) / q.len).toBeLessThan(1e-3);
      });
    }
  }

  it("skips the gradient arrows where the gradient vanishes", () => {
    const s = getSurface("Paraboloid");
    const p = pointGeometry(s.f, 0, 0, 0);
    expect(p.lifted).toBeNull();
    expect(p.floor).toBeNull();
    expect(p.normal.z[0] - p.normal.z[1]).toBeCloseTo(1.6, 12); // vertical normal
  });

  it("clips the tangent lines to the grid at a corner", () => {
    const s = getSurface("Monkey saddle");
    const p = pointGeometry(s.f, 3, -3, surfaceGrid(s).zmin);
    expect(p.tanX.x[0]).toBeCloseTo(2.1, 12);
    expect(p.tanX.x[59]).toBe(3);
    expect(p.tanY.y[0]).toBe(-3);
    expect(p.plane.x).toEqual([p.tanX.x[0], 3]);
  });

  it("fieldArrows gives unit-direction shafts of length 0.2", () => {
    const axis = [0, 1, 2];
    const g = [[3, 3, 3], [3, 3, 3], [3, 3, 3]];
    const z0 = [[4, 4, 4], [4, 4, 4], [4, 4, 4]];
    const a = fieldArrows(axis, g, z0, { density: 3 });
    expect(a.shafts.x.length).toBe(27);
    expect(Math.hypot(a.shafts.x[1] - a.shafts.x[0], a.shafts.y[1] - a.shafts.y[0])).toBeCloseTo(0.2, 6);
    expect(a.shafts.x[1] / 0.2).toBeCloseTo(0.6, 6);
  });

  // Reviewer cases: partial_derivatives(), the normal end (+0.8 along (fx, fy, -1)/|.|),
  // the lifted gradient end (length 0.6) and the right end of the x tangent
  // (half_len 0.9, clipped), from the original Python via ~/miniforge3/bin/python.
  const EXTRA = [
    ["Original (sin/cos + saddle)", 2.7, -2.4, [0.07192638108340205, 1.1433285024047695, 0.86433966254644], [3.2233670434516695, -2.0043421529564966, -0.38583098467007], [2.9738656013458007, -2.192961602065874, 0.5639962279348156], [1.8000000000000003, 3.0, 0.4149249318048327]],
    ["Monkey saddle", -2.95, 0.05, [-1.5390150000000005, 1.5660000599997392, 0.053100000000028125], [-2.2760205003502563, 0.07285332698609623, -1.9693978057642965], [-2.627398298939047, 0.06093879289273369, -1.0332398668799256], [-3.0, -2.0500000000000003, -0.12961494600023538]],
  ];
  for (const [name, x0, y0, zf, nEnd, lEnd, tan] of EXTRA) {
    it(`matches Python near the grid edge: ${name} at (${x0}, ${y0})`, () => {
      const s = getSurface(name);
      const p = pointGeometry(s.f, x0, y0, 0);
      close([p.z0, p.fx, p.fy], zf, 8);
      close([p.normal.x[1], p.normal.y[1], p.normal.z[1]], nEnd, 8);
      close([p.lifted.x[1], p.lifted.y[1], p.lifted.z[1]], lEnd, 8);
      close([p.tanX.x[0], p.tanX.x[59], p.tanX.z[59]], tan, 8);
    });
  }

  it("Sine product corner (-3, 3) has zero gradient: no arrows, as in Python", () => {
    const p = pointGeometry(getSurface("Sine product").f, -3, 3, 0);
    expect(p.z0).toBeCloseTo(-1, 12);
    expect(p.lifted).toBeNull();
    expect(p.floor).toBeNull();
  });

  it("the fixed axis ranges contain everything drawn at any slider point", () => {
    for (const s of CALCULUS_SURFACES) {
      const d = fieldData(s);
      const lo = d.zmin - Z_PAD - 1e-9;
      const hi = d.zmax + Z_PAD + 1e-9;
      const xyMax = 3 + XY_PAD + 1e-9;
      for (let i = 0; i <= 24; i++) {
        for (let j = 0; j <= 24; j++) {
          const p = pointGeometry(s.f, -3 + i / 4, -3 + j / 4, d.zFloor);
          const zs = [...p.tanX.z, ...p.tanY.z, ...p.plane.z.flat(), ...p.normal.z, ...(p.lifted?.z ?? [])];
          const xy = [...p.normal.x, ...p.normal.y, ...(p.lifted ? [...p.lifted.x, ...p.lifted.y] : []), ...(p.floor ? [...p.floor.x, ...p.floor.y] : [])];
          for (const z of zs) expect(z >= lo && z <= hi).toBe(true);
          for (const v of xy) expect(Math.abs(v) <= xyMax).toBe(true);
        }
      }
    }
  });
});
