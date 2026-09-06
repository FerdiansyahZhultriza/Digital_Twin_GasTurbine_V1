import * as THREE from 'three';

// M701F architecture follows Mitsubishi Power's published configuration.
// Airfoils, dimensions and auxiliary details are illustrative, not OEM CAD.
const TAU = Math.PI * 2;
const X_AXIS = new THREE.Vector3(1, 0, 0);
const Y_AXIS = new THREE.Vector3(0, 1, 0);
const identity = new THREE.Quaternion();

function matrix(position, scale = new THREE.Vector3(1, 1, 1), quaternion = identity) {
  return new THREE.Matrix4().compose(position, quaternion, scale);
}

function radial(x, radius, angle) {
  return new THREE.Vector3(x, radius * Math.cos(angle), radius * Math.sin(angle));
}

function axialMatrix(x, radius, angle, scale, rotation = 0) {
  return matrix(radial(x, radius, angle), scale, new THREE.Quaternion().setFromAxisAngle(X_AXIS, rotation));
}

function linkMatrix(start, end, radius, radiusZ = radius) {
  const direction = end.clone().sub(start);
  const length = direction.length();
  return matrix(
    start.clone().add(end).multiplyScalar(0.5),
    new THREE.Vector3(length, radius, radiusZ),
    new THREE.Quaternion().setFromUnitVectors(X_AXIS, direction.divideScalar(length)),
  );
}

// Revolve a cross section around X. Independent cross-section edges preserve
// machined corners while radial segments retain smooth cylindrical normals.
function revolve(profile, segments = 64, start = 0, end = TAU, closeCuts = false) {
  const vertices = [];
  const indices = [];
  for (let edge = 0; edge < profile.length - 1; edge += 1) {
    const offset = vertices.length / 3;
    for (const [x, radius] of [profile[edge], profile[edge + 1]]) {
      for (let segment = 0; segment <= segments; segment += 1) {
        const angle = start + (end - start) * segment / segments;
        vertices.push(x, radius * Math.cos(angle), radius * Math.sin(angle));
      }
    }
    for (let segment = 0; segment < segments; segment += 1) {
      const a = offset + segment;
      const b = a + segments + 1;
      indices.push(a, b + 1, b, a, a + 1, b + 1);
    }
  }
  if (closeCuts && Math.abs(end - start) < TAU - 0.001) {
    // Shapes use x/r coordinates; triangulation also supports tapered profiles.
    const polygon = profile.slice(0, -1).map(([x, radius]) => new THREE.Vector2(x, radius));
    const triangles = THREE.ShapeUtils.triangulateShape(polygon, []);
    for (const [angle, reverse] of [[start, false], [end, true]]) {
      const offset = vertices.length / 3;
      for (const point of polygon) {
        vertices.push(point.x, point.y * Math.cos(angle), point.y * Math.sin(angle));
      }
      for (const [a, b, c] of triangles) {
        // ShapeUtils returns positive x/r winding. Its normal points along +theta.
        indices.push(offset + a, offset + (reverse ? b : c), offset + (reverse ? c : b));
      }
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();
  return geometry;
}

function shell(x0, x1, r0, r1, thickness, start, end, segments = 64) {
  return revolve([
    [x0, r0], [x1, r1], [x1, r1 - thickness],
    [x0, r0 - thickness], [x0, r0],
  ], segments, start, end, true);
}

// A closed cambered airfoil is lofted through five twisted span stations.
// This models solid blades rather than ribbons or triangular line outlines.
function airfoil({ hub, tip, chord, rootAngle, tipAngle, camber = 0.035, thickness = 0.12, lean = 0 }) {
  const pointsPerSide = 12;
  const spanSegments = 4;
  const perimeter = [];
  for (let sample = 0; sample <= pointsPerSide; sample += 1) {
    perimeter.push([(1 - Math.cos(Math.PI * sample / pointsPerSide)) / 2, 1]);
  }
  for (let sample = pointsPerSide - 1; sample > 0; sample -= 1) {
    perimeter.push([(1 - Math.cos(Math.PI * sample / pointsPerSide)) / 2, -1]);
  }
  const vertices = [];
  const indices = [];
  const count = perimeter.length;
  for (let span = 0; span <= spanSegments; span += 1) {
    const t = span / spanSegments;
    const radius = THREE.MathUtils.lerp(hub, tip, t);
    const stagger = THREE.MathUtils.lerp(rootAngle, tipAngle, t);
    const localChord = chord * (1 - 0.19 * t);
    for (const [u, side] of perimeter) {
      const halfThickness = 5 * thickness * (
        0.2969 * Math.sqrt(u) - 0.1260 * u - 0.3516 * u ** 2
        + 0.2843 * u ** 3 - 0.1036 * u ** 4
      );
      const centerline = camber * 4 * u * (1 - u);
      const chordwise = (u - 0.47) * localChord;
      const transverse = (centerline + side * halfThickness) * localChord;
      vertices.push(
        chordwise * Math.cos(stagger) - transverse * Math.sin(stagger) + lean * t * t,
        radius,
        chordwise * Math.sin(stagger) + transverse * Math.cos(stagger),
      );
    }
  }
  for (let span = 0; span < spanSegments; span += 1) {
    for (let point = 0; point < count; point += 1) {
      const next = (point + 1) % count;
      const a = span * count + point;
      const b = span * count + next;
      const c = (span + 1) * count + next;
      const d = (span + 1) * count + point;
      indices.push(a, b, c, a, c, d);
    }
  }
  // Duplicate cap vertices give the root and tip genuinely flat end faces.
  for (const [span, reverse] of [[0, true], [spanSegments, false]]) {
    const offset = vertices.length / 3;
    const polygon = [];
    for (let point = 0; point < count; point += 1) {
      const base = (span * count + point) * 3;
      const x = vertices[base];
      const y = vertices[base + 1];
      const z = vertices[base + 2];
      vertices.push(x, y, z);
      polygon.push(new THREE.Vector2(x, z));
    }
    for (const [a, b, c] of THREE.ShapeUtils.triangulateShape(polygon, [])) {
      indices.push(offset + a, offset + (reverse ? c : b), offset + (reverse ? b : c));
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();
  return geometry;
}

// One combustor outlet transitions from a round basket to a rounded rectangular
// sector of the turbine annulus. All 20 share this geometry and stay stationary.
function transitionDuct() {
  const vertices = [];
  const indices = [];
  const sections = 6;
  const perimeter = 24;
  for (let wall = 0; wall < 2; wall += 1) {
    for (let section = 0; section <= sections; section += 1) {
      const t = section / sections;
      const x = THREE.MathUtils.lerp(1.15, 2.13, t);
      const radius = THREE.MathUtils.lerp(1.13, 0.82, t * t * (3 - 2 * t));
      const radialSize = THREE.MathUtils.lerp(0.146, 0.165, t) - wall * 0.008;
      const tangentSize = THREE.MathUtils.lerp(0.146, 0.116, t) - wall * 0.008;
      const exponent = 1 - 0.40 * t;
      for (let point = 0; point < perimeter; point += 1) {
        const angle = point * TAU / perimeter;
        const cosine = Math.cos(angle);
        const sine = Math.sin(angle);
        vertices.push(
          x,
          radius + Math.sign(cosine) * Math.abs(cosine) ** exponent * radialSize,
          Math.sign(sine) * Math.abs(sine) ** exponent * tangentSize,
        );
      }
    }
  }
  const wallSize = (sections + 1) * perimeter;
  for (let wall = 0; wall < 2; wall += 1) {
    for (let section = 0; section < sections; section += 1) {
      for (let point = 0; point < perimeter; point += 1) {
        const next = (point + 1) % perimeter;
        const a = wall * wallSize + section * perimeter + point;
        const b = wall * wallSize + section * perimeter + next;
        const c = wall * wallSize + (section + 1) * perimeter + next;
        const d = wall * wallSize + (section + 1) * perimeter + point;
        if (wall === 0) indices.push(a, b, c, a, c, d);
        else indices.push(a, c, b, a, d, c);
      }
    }
  }
  for (const section of [0, sections]) {
    for (let point = 0; point < perimeter; point += 1) {
      const next = (point + 1) % perimeter;
      const a = section * perimeter + point;
      const b = section * perimeter + next;
      const c = b + wallSize;
      const d = a + wallSize;
      if (section === 0) indices.push(a, d, c, a, c, b);
      else indices.push(a, b, c, a, c, d);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();
  return geometry;
}

export function buildTurbine() {
  const geometries = new Set();
  const materials = new Set();
  const root = new THREE.Group();
  root.name = 'M701F gas turbine';
  root.userData = { architecture: 'M701F', geometry: 'Illustrative engineering reconstruction; not OEM CAD' };
  const rotor = new THREE.Group();
  rotor.name = 'Common rotating shaft assembly';
  rotor.userData = { role: 'rotor', axis: 'X' };
  root.add(rotor);
  const fixed = new THREE.Group();
  fixed.name = 'Stationary internal assembly';
  root.add(fixed);
  const casing = new THREE.Group();
  casing.name = 'Removable upper casing';
  casing.userData = { role: 'upperCasing', splitPlane: 'Y = 0' };
  casing.visible = false;
  root.add(casing);
  const lower = new THREE.Group();
  lower.name = 'Lower casing and split flanges';
  root.add(lower);

  function material(color, metalness, roughness) {
    const value = new THREE.MeshStandardMaterial({ color, metalness, roughness });
    materials.add(value);
    return value;
  }
  const steel = material('#a8b0b3', 0.92, 0.29);
  const bladeSteel = material('#b8c3c7', 0.90, 0.28);
  const vaneSteel = material('#7f939b', 0.87, 0.39);
  const machined = material('#cbd0cc', 0.92, 0.25);
  const drumMetal = material('#667b82', 0.88, 0.35);
  const hotMetal = material('#aaa08b', 0.88, 0.36);
  const hotVanes = material('#858171', 0.84, 0.44);
  const combustorMetal = material('#8d8a7d', 0.88, 0.40);
  const transitionMetal = material('#837664', 0.86, 0.47);
  const castMetal = material('#29414b', 0.70, 0.52);
  const castRibs = material('#40545b', 0.78, 0.43);
  const boltsMetal = material('#849298', 0.91, 0.28);
  const darkMetal = material('#303b40', 0.78, 0.46);
  const gasket = material('#19262b', 0.40, 0.72);
  const baseMetal = material('#25383e', 0.65, 0.56);

  function own(geometry) {
    geometries.add(geometry);
    return geometry;
  }
  function mesh(geometry, surface, name, parent = fixed) {
    const value = new THREE.Mesh(own(geometry), surface);
    value.name = name;
    value.castShadow = true;
    value.receiveShadow = true;
    parent.add(value);
    return value;
  }
  function instances(geometry, surface, transforms, name, parent = fixed, userData = {}) {
    const value = new THREE.InstancedMesh(own(geometry), surface, transforms.length);
    transforms.forEach((transform, index) => value.setMatrixAt(index, transform));
    value.instanceMatrix.needsUpdate = true;
    value.name = name;
    value.userData = userData;
    value.castShadow = true;
    value.receiveShadow = true;
    value.computeBoundingSphere();
    parent.add(value);
    return value;
  }

  const box = own(new THREE.BoxGeometry(1, 1, 1));
  const cylinder = own(new THREE.CylinderGeometry(1, 1, 1, 20, 1, false).rotateZ(-Math.PI / 2));
  const hex = own(new THREE.CylinderGeometry(1, 1, 1, 6, 1, false).rotateZ(-Math.PI / 2));
  const band = own(shell(-0.5, 0.5, 1, 1, 0.10, 0, TAU, 32));
  const ring = own(shell(-0.5, 0.5, 1, 1, 0.055, 0, TAU, 64));
  const shaft = mesh(cylinder, machined, 'Continuous shaft', rotor);
  shaft.scale.set(12.5, 0.185, 0.185);
  shaft.position.x = -0.02;
  mesh(revolve([
    [-4.83, 0.185], [-4.83, 0.46], [-4.61, 0.486],
    [-3.10, 0.567], [-1.58, 0.644], [-0.48, 0.703],
    [-0.35, 0.64], [-0.35, 0.185], [-4.83, 0.185],
  ]), drumMetal, 'Bolted compressor rotor drum', rotor);
  mesh(revolve([
    [-0.34, 0.185], [-0.34, 0.57], [0.1, 0.50], [0.45, 0.33],
    [1.70, 0.33], [2.22, 0.57], [2.22, 0.185], [-0.34, 0.185],
  ]), steel, 'Central torque shaft and rotor couplings', rotor);

  const compressorRimTransforms = [];
  const compressorHubTransforms = [];
  const statorRingTransforms = [];
  for (let stage = 0; stage < 17; stage += 1) {
    const x = -4.65 + stage * 0.245;
    const hub = 0.505 + stage * 0.0125;
    const tip = 1.125 - stage * 0.012;
    const bladeCount = 48 + stage * 2;
    const rotorTransforms = [];
    for (let blade = 0; blade < bladeCount; blade += 1) {
      rotorTransforms.push(matrix(
        new THREE.Vector3(x, 0, 0),
        new THREE.Vector3(1, 1, 1),
        new THREE.Quaternion().setFromAxisAngle(X_AXIS, blade * TAU / bladeCount + stage * 0.031),
      ));
    }
    instances(airfoil({
      hub, tip, chord: 0.163 - stage * 0.0011,
      rootAngle: 0.73, tipAngle: 0.43, camber: 0.045, thickness: 0.13, lean: 0.016,
    }), bladeSteel, rotorTransforms, `Compressor rotor stage ${String(stage + 1).padStart(2, '0')}`, rotor,
    { role: 'compressorRotor', stage: stage + 1, bladeCount });
    const statorCount = 52 + stage * 2;
    const statorTransforms = [];
    for (let vane = 0; vane < statorCount; vane += 1) {
      statorTransforms.push(matrix(
        new THREE.Vector3(x + 0.121, 0, 0),
        new THREE.Vector3(1, 1, 1),
        new THREE.Quaternion().setFromAxisAngle(X_AXIS, vane * TAU / statorCount),
      ));
    }
    instances(airfoil({
      hub: hub + 0.022, tip: tip + 0.008, chord: 0.129 - stage * 0.0005,
      rootAngle: -0.58, tipAngle: -0.39, camber: -0.04, thickness: 0.13,
    }), vaneSteel, statorTransforms, `Compressor stator row ${String(stage + 1).padStart(2, '0')}`, fixed,
    { role: 'compressorStator', stage: stage + 1, vaneCount: statorCount });
    compressorRimTransforms.push(matrix(new THREE.Vector3(x, 0, 0), new THREE.Vector3(0.038, hub + 0.007, hub + 0.007)));
    compressorHubTransforms.push(matrix(new THREE.Vector3(x, 0, 0), new THREE.Vector3(0.087, hub, hub)));
    statorRingTransforms.push(matrix(new THREE.Vector3(x + 0.121, 0, 0), new THREE.Vector3(0.028, tip + 0.024, tip + 0.024)));
  }
  instances(ring, machined, compressorRimTransforms, '17 machined compressor disk rims', rotor);
  instances(cylinder, drumMetal, compressorHubTransforms, '17 compressor disk bodies', rotor);
  // Aft half of stator retaining rings: enough to show their casing attachment
  // without rings crossing the open inspection area.
  instances(shell(-0.5, 0.5, 1, 1, 0.028, Math.PI / 2, Math.PI * 1.5, 48),
    castRibs, statorRingTransforms, 'Compressor lower stator retaining rings', lower);

  const guideVanes = [];
  const guidePivots = [];
  for (let vane = 0; vane < 44; vane += 1) {
    const angle = vane * TAU / 44;
    guideVanes.push(matrix(new THREE.Vector3(-4.89, 0, 0), new THREE.Vector3(1, 1, 1),
      new THREE.Quaternion().setFromAxisAngle(X_AXIS, angle)));
    guidePivots.push(linkMatrix(radial(-4.89, 1.13, angle), radial(-4.89, 1.205, angle), 0.026));
  }
  instances(airfoil({ hub: 0.47, tip: 1.155, chord: 0.14, rootAngle: -0.24, tipAngle: -0.16, camber: -0.02 }),
    steel, guideVanes, '44 variable inlet guide vanes', fixed, { role: 'inletGuideVanes', vaneCount: 44 });
  instances(cylinder, boltsMetal, guidePivots, 'Inlet guide vane pivots');

  const turbineRims = [];
  const turbineDisks = [];
  const turbineShrouds = [];
  const turbineStatorRings = [];
  for (let stage = 0; stage < 4; stage += 1) {
    const x = 2.47 + stage * 0.64;
    const hub = 0.59 - stage * 0.02;
    const tip = 0.96 + stage * 0.112;
    const bladeCount = 58 + stage * 6;
    const transforms = [];
    for (let blade = 0; blade < bladeCount; blade += 1) {
      transforms.push(matrix(new THREE.Vector3(x, 0, 0), new THREE.Vector3(1, 1, 1),
        new THREE.Quaternion().setFromAxisAngle(X_AXIS, blade * TAU / bladeCount + stage * 0.032)));
    }
    instances(airfoil({
      hub, tip, chord: 0.31 + stage * 0.018, rootAngle: -0.49, tipAngle: -0.90,
      camber: 0.095, thickness: 0.17, lean: -0.013,
    }), stage < 2 ? hotMetal : steel, transforms, `Turbine rotor stage ${stage + 1}`, rotor,
    { role: 'turbineRotor', stage: stage + 1, bladeCount, shrouded: stage >= 2 });
    const vaneCount = 42 + stage * 4;
    const nozzles = [];
    for (let vane = 0; vane < vaneCount; vane += 1) {
      nozzles.push(matrix(new THREE.Vector3(x - 0.30, 0, 0), new THREE.Vector3(1, 1, 1),
        new THREE.Quaternion().setFromAxisAngle(X_AXIS, vane * TAU / vaneCount)));
    }
    instances(airfoil({
      hub: hub + 0.012, tip: tip + 0.017, chord: 0.265 + stage * 0.011,
      rootAngle: 0.58, tipAngle: 0.84, camber: -0.10, thickness: 0.17,
    }), hotVanes, nozzles, `Turbine nozzle guide vane row ${stage + 1}`, fixed,
    { role: 'turbineNozzle', stage: stage + 1, vaneCount });
    turbineDisks.push(matrix(new THREE.Vector3(x, 0, 0), new THREE.Vector3(0.24, hub, hub)));
    turbineRims.push(matrix(new THREE.Vector3(x, 0, 0), new THREE.Vector3(0.25, hub + 0.012, hub + 0.012)));
    turbineStatorRings.push(matrix(new THREE.Vector3(x - 0.30, 0, 0), new THREE.Vector3(0.045, tip + 0.035, tip + 0.035)));
    if (stage >= 2) {
      turbineShrouds.push(matrix(new THREE.Vector3(x - 0.02, 0, 0), new THREE.Vector3(0.19, tip + 0.009, tip + 0.009)));
    }
  }
  instances(cylinder, drumMetal, turbineDisks, 'Four turbine rotor disks', rotor);
  instances(ring, machined, turbineRims, 'Four turbine disk platforms', rotor);
  instances(shell(-0.5, 0.5, 1, 1, 0.015, 0, TAU, 96), hotMetal,
    turbineShrouds, 'Integral shrouds on turbine stages 3 and 4', rotor, { role: 'turbineTipShrouds', stages: [3, 4] });
  instances(shell(-0.5, 0.5, 1, 1, 0.03, Math.PI / 2, Math.PI * 1.5, 48), hotVanes,
    turbineStatorRings, 'Independent lower turbine vane carrier rings', lower);
  mesh(revolve([
    [4.53, 0.185], [4.53, 0.53], [4.83, 0.45], [5.3, 0.35],
    [5.90, 0.245], [5.98, 0.185], [4.53, 0.185],
  ]), steel, 'Axial exhaust inner diffuser cone');

  const baskets = [];
  const basketBands = [];
  const burnerCovers = [];
  const burnerBolts = [];
  const nozzles = [];
  const pipes = [];
  const ducts = [];
  const basketGeometries = own(new THREE.CylinderGeometry(0.84, 1, 1, 32, 1, false).rotateZ(-Math.PI / 2));
  for (let can = 0; can < 20; can += 1) {
    const angle = can * TAU / 20 + Math.PI / 20;
    const head = radial(-0.055, 1.265, angle);
    const tail = radial(1.23, 1.123, angle);
    const direction = tail.clone().sub(head).normalize();
    const orientation = new THREE.Quaternion().setFromUnitVectors(X_AXIS, direction);
    baskets.push(linkMatrix(head, tail, 0.171));
    for (const t of [0.025, 0.32, 0.69, 0.94]) {
      const center = head.clone().lerp(tail, t);
      const radius = THREE.MathUtils.lerp(0.179, 0.151, t);
      basketBands.push(matrix(center, new THREE.Vector3(0.038, radius, radius), orientation));
    }
    const coverCenter = head.clone().addScaledVector(direction, -0.039);
    burnerCovers.push(matrix(coverCenter, new THREE.Vector3(0.063, 0.186, 0.186), orientation));
    for (let bolt = 0; bolt < 10; bolt += 1) {
      const boltAngle = bolt * TAU / 10;
      const offset = new THREE.Vector3(-0.047, 0.161 * Math.cos(boltAngle), 0.161 * Math.sin(boltAngle)).applyQuaternion(orientation);
      burnerBolts.push(matrix(coverCenter.clone().add(offset), new THREE.Vector3(0.033, 0.014, 0.014), orientation));
    }
    // One pilot plus eight surrounding main burner fittings.
    for (let burner = 0; burner < 9; burner += 1) {
      const burnerAngle = (burner - 1) * TAU / 8;
      const radius = burner === 0 ? 0 : 0.100;
      const offset = new THREE.Vector3(-0.052, radius * Math.cos(burnerAngle), radius * Math.sin(burnerAngle)).applyQuaternion(orientation);
      nozzles.push(matrix(coverCenter.clone().add(offset), new THREE.Vector3(0.063, burner === 0 ? 0.032 : 0.023, burner === 0 ? 0.032 : 0.023), orientation));
    }
    const feedStart = radial(-0.33, 1.395, angle);
    const feedBend = radial(-0.33, 1.265, angle);
    const feedEnd = head.clone().addScaledVector(direction, -0.14);
    pipes.push(linkMatrix(feedStart, feedBend, 0.018), linkMatrix(feedBend, feedEnd, 0.018));
    ducts.push(matrix(new THREE.Vector3(), new THREE.Vector3(1, 1, 1), new THREE.Quaternion().setFromAxisAngle(X_AXIS, angle)));
  }
  instances(basketGeometries, combustorMetal, baskets, '20 can-annular combustor baskets', fixed, { role: 'combustors', canCount: 20 });
  instances(band, steel, basketBands, 'Combustor cooling sleeve bands');
  instances(cylinder, darkMetal, burnerCovers, '20 bolted burner end covers');
  instances(hex, boltsMetal, burnerBolts, 'Combustor cover fasteners');
  instances(cylinder, steel, nozzles, 'Pilot and main burner fittings', fixed, { role: 'burnerFittings', mainBurnersPerCan: 8, pilotBurnersPerCan: 1 });
  instances(cylinder, steel, pipes, 'Fuel manifold branch pipes');
  instances(transitionDuct(), transitionMetal, ducts, '20 shaped combustor transition ducts', fixed, { role: 'transitionDucts', count: 20 });
  const manifold = mesh(new THREE.TorusGeometry(1.395, 0.026, 10, 128).rotateY(Math.PI / 2), steel, 'Circumferential fuel manifold');
  manifold.position.x = -0.33;

  const panelDefinitions = [
    [-5.31, -4.78, 1.365, 1.220, 'Inlet bellmouth'],
    [-4.78, -3.40, 1.220, 1.151, 'Forward compressor casing'],
    [-3.40, -1.98, 1.151, 1.080, 'Middle compressor casing'],
    [-1.98, -0.48, 1.080, 1.005, 'Aft compressor casing'],
    [-0.48, -0.32, 1.005, 1.490, 'Compressor discharge casing'],
    [-0.32, 1.97, 1.490, 1.460, 'Combustor pressure casing'],
    [1.97, 3.28, 1.078, 1.235, 'Forward turbine casing'],
    [3.28, 4.72, 1.235, 1.430, 'Aft turbine casing'],
    [4.72, 6.17, 1.430, 1.560, 'Exhaust diffuser casing'],
  ];
  const seamLower = [];
  const seamUpper = [];
  const seamBoltsLower = [];
  const seamBoltsUpper = [];
  const seamBoltOrientation = new THREE.Quaternion().setFromUnitVectors(X_AXIS, Y_AXIS);
  for (const [x0, x1, r0, r1, name] of panelDefinitions) {
    mesh(shell(x0, x1, r0, r1, 0.055, Math.PI / 2, Math.PI * 1.5, 64), castMetal, `${name}: lower half`, lower);
    mesh(shell(x0, x1, r0, r1, 0.055, -Math.PI / 2, Math.PI / 2, 64), castMetal, `${name}: upper half`, casing);
    if (x1 - x0 < 0.2) continue;
    for (const side of [-1, 1]) {
      const radius = (r0 + r1) / 2 + 0.025;
      const slope = Math.atan2((r1 - r0) * side, x1 - x0);
      const orientation = new THREE.Quaternion().setFromAxisAngle(Y_AXIS, -slope);
      const length = Math.hypot(x1 - x0, r1 - r0);
      seamLower.push(matrix(new THREE.Vector3((x0 + x1) / 2, -0.039, side * radius), new THREE.Vector3(length, 0.078, 0.12), orientation));
      seamUpper.push(matrix(new THREE.Vector3((x0 + x1) / 2, 0.039, side * radius), new THREE.Vector3(length, 0.078, 0.12), orientation));
      const boltCount = Math.max(3, Math.floor(length / 0.22));
      for (let bolt = 0; bolt < boltCount; bolt += 1) {
        const t = (bolt + 0.5) / boltCount;
        const x = THREE.MathUtils.lerp(x0, x1, t);
        const z = side * (THREE.MathUtils.lerp(r0, r1, t) + 0.025);
        seamBoltsLower.push(matrix(new THREE.Vector3(x, -0.093, z), new THREE.Vector3(0.044, 0.030, 0.030), seamBoltOrientation));
        seamBoltsUpper.push(matrix(new THREE.Vector3(x, 0.097, z), new THREE.Vector3(0.044, 0.031, 0.031), seamBoltOrientation));
      }
    }
  }
  instances(box, castRibs, seamLower, 'Machined horizontal lower split flanges', lower);
  instances(box, castRibs, seamUpper, 'Upper horizontal split flanges', casing);
  instances(hex, boltsMetal, seamBoltsLower, 'Lower horizontal split flange nuts', lower);
  instances(hex, boltsMetal, seamBoltsUpper, 'Upper horizontal split flange bolt heads', casing);

  const flangeDefinitions = [
    [-5.31, 1.385], [-4.78, 1.248], [-3.40, 1.183], [-1.98, 1.111],
    [-0.32, 1.527], [1.97, 1.496], [3.28, 1.270], [4.72, 1.469], [6.17, 1.602],
  ];
  const flangeLower = [];
  const flangeUpper = [];
  const flangeBoltsLower = [];
  const flangeBoltsUpper = [];
  const ribLower = [];
  const ribUpper = [];
  for (const [x, radius] of flangeDefinitions) {
    const transform = matrix(new THREE.Vector3(x, 0, 0), new THREE.Vector3(0.107, radius, radius));
    flangeLower.push(transform);
    flangeUpper.push(transform);
    const count = 48;
    for (let bolt = 0; bolt < count; bolt += 1) {
      const angle = (bolt + 0.5) * TAU / count;
      const target = Math.cos(angle) > 0 ? flangeBoltsUpper : flangeBoltsLower;
      target.push(axialMatrix(x - 0.072, radius - 0.038, angle, new THREE.Vector3(0.035, 0.029, 0.029)));
      target.push(axialMatrix(x + 0.072, radius - 0.038, angle, new THREE.Vector3(0.035, 0.029, 0.029)));
    }
  }
  for (const [x0, x1, r0, r1] of panelDefinitions) {
    if (x1 - x0 < 0.5) continue;
    for (let rib = 1; rib <= 3; rib += 1) {
      const t = rib / 4;
      const radius = THREE.MathUtils.lerp(r0, r1, t) + 0.021;
      const transform = matrix(new THREE.Vector3(THREE.MathUtils.lerp(x0, x1, t), 0, 0), new THREE.Vector3(0.035, radius, radius));
      ribLower.push(transform);
      ribUpper.push(transform);
    }
  }
  const lowerRing = own(shell(-0.5, 0.5, 1, 1, 0.075, Math.PI / 2, Math.PI * 1.5, 64));
  const upperRing = own(shell(-0.5, 0.5, 1, 1, 0.075, -Math.PI / 2, Math.PI / 2, 64));
  instances(lowerRing, castRibs, flangeLower, 'Bolted circumferential lower casing flanges', lower);
  instances(upperRing, castRibs, flangeUpper, 'Bolted circumferential upper casing flanges', casing);
  instances(lowerRing, castRibs, ribLower, 'Lower casing stiffening ribs', lower);
  instances(upperRing, castRibs, ribUpper, 'Upper casing stiffening ribs', casing);
  instances(hex, boltsMetal, flangeBoltsLower, 'Lower circumferential flange bolts', lower);
  instances(hex, boltsMetal, flangeBoltsUpper, 'Upper circumferential flange bolts', casing);

  // The M701F rotor uses two bearing supports. Dataset channel labels are not
  // interpreted as additional physical bearings in this reconstruction.
  const bearingBodies = [];
  const bearingCollars = [];
  const bearingBases = [];
  const bearingCaps = [];
  const bearingFasteners = [];
  const oilLines = [];
  for (const [index, x] of [-5.66, 5.75].entries()) {
    const bearing = new THREE.Group();
    bearing.name = index === 0 ? 'Compressor-end bearing support' : 'Exhaust-end bearing support';
    bearing.userData = { role: 'bearing', position: index === 0 ? 'coldEnd' : 'exhaustEnd' };
    fixed.add(bearing);
    bearing.position.x = x;
    bearingBodies.push(matrix(new THREE.Vector3(x, 0, 0), new THREE.Vector3(0.48, 0.37, 0.37)));
    bearingCollars.push(matrix(new THREE.Vector3(x - 0.25, 0, 0), new THREE.Vector3(0.072, 0.405, 0.405)));
    bearingCollars.push(matrix(new THREE.Vector3(x + 0.25, 0, 0), new THREE.Vector3(0.072, 0.405, 0.405)));
    bearingBases.push(matrix(new THREE.Vector3(x, -0.99, 0), new THREE.Vector3(0.64, 1.18, 0.66)));
    bearingCaps.push(matrix(new THREE.Vector3(x, -0.42, 0), new THREE.Vector3(0.77, 0.10, 0.88)));
    for (const dx of [-0.25, 0.25]) {
      for (const z of [-0.33, 0.33]) {
        bearingFasteners.push(matrix(new THREE.Vector3(x + dx, -0.342, z), new THREE.Vector3(0.058, 0.047, 0.047), seamBoltOrientation));
      }
    }
    oilLines.push(linkMatrix(new THREE.Vector3(x + 0.05, -1.52, 0.46), new THREE.Vector3(x + 0.05, -0.08, 0.46), 0.027));
    oilLines.push(linkMatrix(new THREE.Vector3(x + 0.05, -0.08, 0.46), new THREE.Vector3(x + 0.05, -0.08, 0.34), 0.027));
  }
  instances(shell(-0.5, 0.5, 1, 1, 0.46, 0, TAU, 64), castRibs, bearingBodies, 'Two journal bearing housings');
  instances(band, machined, bearingCollars, 'Bearing labyrinth seal collars');
  instances(box, castMetal, bearingBases, 'Two bearing pedestals');
  instances(box, castRibs, bearingCaps, 'Bearing pedestal mounting plates');
  instances(hex, boltsMetal, bearingFasteners, 'Bearing pedestal fasteners');
  instances(cylinder, steel, oilLines, 'Bearing oil supply tubes');

  const coupling = [];
  for (const [x, length, radius] of [[-6.13, 0.12, 0.33], [-6.00, 0.08, 0.32], [-6.31, 0.14, 0.225]]) {
    coupling.push(matrix(new THREE.Vector3(x, 0, 0), new THREE.Vector3(length, radius, radius)));
  }
  instances(cylinder, machined, coupling, 'Cold-end generator coupling flanges', rotor, { role: 'outputCoupling', location: 'compressorEnd' });
  const couplingBolts = [];
  for (let bolt = 0; bolt < 12; bolt += 1) {
    couplingBolts.push(axialMatrix(-6.218, 0.275, bolt * TAU / 12, new THREE.Vector3(0.058, 0.036, 0.036)));
  }
  instances(hex, boltsMetal, couplingBolts, 'Cold-end coupling bolts', rotor);

  const exhaustStruts = [];
  for (let strut = 0; strut < 6; strut += 1) {
    const angle = strut * TAU / 6 + Math.PI / 6;
    exhaustStruts.push(linkMatrix(radial(5.79, 0.39, angle), radial(5.92, 1.475, angle), 0.063, 0.092));
  }
  instances(cylinder, castRibs, exhaustStruts, 'Six exhaust bearing support struts');

  const skidBeams = [];
  const skidFeet = [];
  const casingFeet = [];
  const foundationBolts = [];
  for (const z of [-1.09, 1.09]) {
    skidBeams.push(matrix(new THREE.Vector3(0.0, -1.72, z), new THREE.Vector3(12.7, 0.19, 0.24)));
    for (const x of [-5.65, -3.55, -0.1, 3.45, 5.75]) {
      skidFeet.push(matrix(new THREE.Vector3(x, -1.828, z), new THREE.Vector3(0.88, 0.08, 0.58)));
      for (const dx of [-0.29, 0.29]) {
        foundationBolts.push(matrix(new THREE.Vector3(x + dx, -1.756, z + 0.19), new THREE.Vector3(0.060, 0.044, 0.044), seamBoltOrientation));
        foundationBolts.push(matrix(new THREE.Vector3(x + dx, -1.756, z - 0.19), new THREE.Vector3(0.060, 0.044, 0.044), seamBoltOrientation));
      }
    }
    for (const x of [-3.4, -0.14, 3.30, 4.78]) {
      casingFeet.push(matrix(new THREE.Vector3(x, -1.35, z * 0.82), new THREE.Vector3(0.30, 0.57, 0.29)));
    }
  }
  for (const x of [-5.66, -3.4, -0.14, 3.3, 5.75]) {
    skidBeams.push(matrix(new THREE.Vector3(x, -1.68, 0), new THREE.Vector3(0.29, 0.21, 2.35)));
  }
  instances(box, baseMetal, skidBeams, 'Fabricated twin longitudinal mounting skids');
  instances(box, darkMetal, skidFeet, 'Base sole plates');
  instances(box, castRibs, casingFeet, 'Casing mounting feet');
  instances(hex, boltsMetal, foundationBolts, 'Foundation anchor nuts');

  // Small service features ground the casting in an industrial assembly.
  const liftingEyes = [];
  const serviceBosses = [];
  for (const [x, radius] of [[-3.70, 1.205], [-1.18, 1.08], [0.25, 1.49], [1.54, 1.475], [3.69, 1.32]]) {
    liftingEyes.push(matrix(new THREE.Vector3(x, radius + 0.045, 0), new THREE.Vector3(0.115, 0.115, 0.115),
      new THREE.Quaternion().setFromAxisAngle(Y_AXIS, Math.PI / 2)));
    serviceBosses.push(linkMatrix(radial(x, radius - 0.025, 0.54), radial(x, radius + 0.065, 0.54), 0.065));
  }
  instances(new THREE.TorusGeometry(1, 0.25, 8, 24), castRibs, liftingEyes, 'Upper casing lifting lugs', casing);
  instances(cylinder, boltsMetal, serviceBosses, 'Upper casing inspection bosses', casing);
  // Thin dark split-line strips are physically separate from removable covers.
  const labelPlate = mesh(box, gasket, 'Compressor casting identification plate', lower);
  labelPlate.position.set(-3.83, -0.29, 1.155);
  labelPlate.scale.set(0.45, 0.15, 0.015);

  const stats = { compressorStages: 17, combustors: 20, turbineStages: 4, bearings: 2 };
  root.userData.stats = { ...stats };
  const labels = [
    { text: 'COLD-END DRIVE', position: new THREE.Vector3(-5.89, 0.57, 0.18), detail: 'Generator coupling at the compressor end' },
    { text: 'AXIAL COMPRESSOR', position: new THREE.Vector3(-3.10, 1.35, 0.20), detail: '17 rotor stages · stationary guide vanes' },
    { text: 'CAN-ANNULAR COMBUSTION', position: new THREE.Vector3(0.62, 1.70, 0.20), detail: '20 combustors · individual transition ducts' },
    { text: 'POWER TURBINE', position: new THREE.Vector3(3.34, 1.44, 0.20), detail: '4 rotor stages · 4 upstream nozzle rows' },
    { text: 'AXIAL EXHAUST', position: new THREE.Vector3(5.63, 1.67, 0.20), detail: 'Annular diffuser and exhaust-end bearing' },
  ];
  root.updateMatrixWorld(true);
  return {
    root, rotor, casing, labels, stats,
    dispose() {
      geometries.forEach((geometry) => geometry.dispose());
      materials.forEach((surface) => surface.dispose());
    },
  };
}
