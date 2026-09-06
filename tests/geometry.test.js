import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import { buildTurbine } from '../src/viewer/geometry.js';
import { rotationStep } from '../src/viewer/viewer.js';

const model = buildTurbine();
const meshes = [];
model.root.traverse((object) => { if (object.isMesh) meshes.push(object); });

test('published architecture is represented by actual stationary and rotating rows', () => {
  assert.deepEqual(model.stats, { compressorStages: 17, combustors: 20, turbineStages: 4, bearings: 2 });
  for (const [role, expected, parent] of [
    ['compressorRotor', 17, model.rotor], ['compressorStator', 17, null],
    ['turbineRotor', 4, model.rotor], ['turbineNozzle', 4, null],
  ]) {
    const rows = meshes.filter((object) => object.userData.role === role);
    assert.equal(rows.length, expected, role);
    assert.deepEqual(rows.map((row) => row.userData.stage), Array.from({ length: expected }, (_, i) => i + 1));
    for (const row of rows) {
      assert.ok(row.isInstancedMesh && row.count >= 40, `${row.name} must contain solid blade instances`);
      if (parent) assert.equal(row.parent, parent);
      else assert.notEqual(row.parent, model.rotor);
    }
  }
  assert.equal(meshes.find((object) => object.userData.role === 'combustors').count, 20);
  assert.equal(meshes.find((object) => object.userData.role === 'transitionDucts').count, 20);
  assert.equal(meshes.find((object) => object.userData.role === 'inletGuideVanes').count, 44);
  const bearings = [];
  model.root.traverse((object) => { if (object.userData.role === 'bearing') bearings.push(object); });
  assert.equal(bearings.length, 2);
  assert.deepEqual(meshes.filter((object) => object.userData.role === 'turbineRotor' && object.userData.shrouded).map((row) => row.userData.stage), [3, 4]);
});

test('all geometry has finite positions, useful normals and valid triangle indices', () => {
  const checked = new Set();
  for (const object of meshes) {
    const geometry = object.geometry;
    if (checked.has(geometry)) continue;
    checked.add(geometry);
    const positions = geometry.getAttribute('position');
    const normals = geometry.getAttribute('normal');
    assert.ok(positions.count > 0, object.name);
    assert.equal(normals.count, positions.count, object.name);
    for (let vertex = 0; vertex < positions.count; vertex += 1) {
      const point = new THREE.Vector3().fromBufferAttribute(positions, vertex);
      const normal = new THREE.Vector3().fromBufferAttribute(normals, vertex);
      assert.ok(point.toArray().every(Number.isFinite), `${object.name}: position ${vertex}`);
      assert.ok(normal.toArray().every(Number.isFinite), `${object.name}: normal ${vertex}`);
      assert.ok(normal.length() > 0.99 && normal.length() < 1.01, `${object.name}: unit normal ${vertex}`);
    }
    const indices = geometry.getIndex();
    assert.equal(indices.count % 3, 0, object.name);
    for (const index of indices.array) assert.ok(Number.isInteger(index) && index >= 0 && index < positions.count, object.name);
  }
  for (const object of meshes.filter((entry) => entry.isInstancedMesh)) {
    assert.ok(Array.from(object.instanceMatrix.array).every(Number.isFinite), `${object.name}: finite instance matrices`);
  }
});

test('airfoil tip and root caps point outwards, casing outer normals face away from shaft', () => {
  const blade = meshes.find((object) => object.userData.role === 'compressorRotor');
  const position = blade.geometry.getAttribute('position');
  const normal = blade.geometry.getAttribute('normal');
  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < position.count; i += 1) {
    minY = Math.min(minY, position.getY(i));
    maxY = Math.max(maxY, position.getY(i));
  }
  let rootFaces = 0;
  let tipFaces = 0;
  for (let i = 0; i < position.count; i += 1) {
    if (Math.abs(normal.getY(i)) < 0.99) continue;
    if (Math.abs(position.getY(i) - minY) < 1e-5) {
      assert.ok(normal.getY(i) < -0.99, 'blade root cap normal');
      rootFaces += 1;
    }
    if (Math.abs(position.getY(i) - maxY) < 1e-5) {
      assert.ok(normal.getY(i) > 0.99, 'blade tip cap normal');
      tipFaces += 1;
    }
  }
  assert.ok(rootFaces >= 12 && tipFaces >= 12);
  const shell = meshes.find((object) => object.name === 'Middle compressor casing: lower half');
  const shellPosition = shell.geometry.getAttribute('position');
  const shellNormal = shell.geometry.getAttribute('normal');
  // The first surface is the exterior profile edge, with 65 angular vertices.
  for (let i = 0; i < 65; i += 1) {
    const radialDot = shellPosition.getY(i) * shellNormal.getY(i) + shellPosition.getZ(i) * shellNormal.getZ(i);
    assert.ok(radialDot > 1, `casing outer normal ${i}`);
  }
});

test('bounds, draw calls and separately removable casing remain practical', () => {
  model.root.updateMatrixWorld(true);
  const bounds = new THREE.Box3().setFromObject(model.root);
  assert.ok(bounds.min.x >= -6.5 && bounds.max.x <= 6.5, `axial bounds: ${bounds.min.x}..${bounds.max.x}`);
  assert.ok(bounds.min.y > -1.95 && bounds.max.y < 1.95, `vertical bounds: ${bounds.min.y}..${bounds.max.y}`);
  assert.ok(bounds.min.z > -1.8 && bounds.max.z < 1.8, `lateral bounds: ${bounds.min.z}..${bounds.max.z}`);
  assert.ok(meshes.length < 200, `${meshes.length} mesh draw calls`);
  assert.equal(model.casing.visible, false);
  model.casing.visible = true;
  assert.equal(model.rotor.visible, true);
  assert.equal(model.root.getObjectByName('Lower casing and split flanges').visible, true);
  const upperBounds = new THREE.Box3().setFromObject(model.casing);
  assert.ok(upperBounds.min.y > -1e-5, `upper casing below split plane: ${upperBounds.min.y}`);
  model.casing.visible = false;
});

test('one X-axis shaft rotation moves every rotor while stationary rows and casing stay fixed', () => {
  const rotorRow = meshes.find((object) => object.userData.role === 'compressorRotor');
  const hotRow = meshes.find((object) => object.userData.role === 'turbineRotor');
  const stator = meshes.find((object) => object.userData.role === 'compressorStator');
  const fixedBefore = stator.matrixWorld.clone();
  const casingBefore = model.casing.matrixWorld.clone();
  const coldBefore = rotorRow.matrixWorld.clone();
  const hotBefore = hotRow.matrixWorld.clone();
  model.rotor.rotation.x = 0.31;
  model.root.updateMatrixWorld(true);
  assert.ok(!rotorRow.matrixWorld.equals(coldBefore));
  assert.ok(!hotRow.matrixWorld.equals(hotBefore));
  assert.ok(stator.matrixWorld.equals(fixedBefore));
  assert.ok(model.casing.matrixWorld.equals(casingBefore));
  assert.equal(model.rotor.rotation.y, 0);
  assert.equal(model.rotor.rotation.z, 0);
  model.rotor.rotation.x = 0;
  model.root.updateMatrixWorld(true);
});

test('rotation stops for missing or invalid speed and scales proportionally with valid rpm', () => {
  for (const rpm of [0, -3000, NaN, Infinity, -Infinity, null, undefined]) assert.equal(rotationStep(rpm, 0.016), 0);
  assert.equal(rotationStep(3000, 0), 0);
  assert.equal(rotationStep(3000, -1), 0);
  assert.ok(rotationStep(3000, 0.016) > 0);
  assert.equal(rotationStep(3000, 0.016), 2 * rotationStep(1500, 0.016));
  assert.equal(rotationStep(3000, 100), rotationStep(3000, 0.05));
  assert.equal(rotationStep(99999, 0.016), rotationStep(12000, 0.016));
});

test.after(() => model.dispose());
