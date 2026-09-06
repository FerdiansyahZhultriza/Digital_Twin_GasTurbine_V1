import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { buildTurbine } from './geometry.js';

// Keep an instance through Streamlit's data-update cleanup/setup cycle. An actual
// unmount disposes its WebGL context, observers, geometry and event listeners.
const instances = new WeakMap();
const TAU = Math.PI * 2;

export function rotationStep(rpm, seconds) {
  if (rpm === null || !Number.isFinite(rpm) || rpm <= 0) return 0;
  return Math.min(rpm, 12000) / 600 * TAU / 60 * Math.min(Math.max(seconds, 0), .05);
}

function createViewer(element, initialData) {
  const viewport = element.querySelector('.viewport');
  const annotations = element.querySelector('.annotations');
  const abort = new AbortController();
  let renderer, model, controls, environmentTarget, shadowTexture;
  let resizeObserver, intersectionObserver, animationId, stopped = false, visible = true;
  let latestData = initialData;
  const state = { rpm:null, playing:!matchMedia('(prefers-reduced-motion: reduce)').matches, labels:false, mode:'cutaway' };
  const labelElements = [];
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(33, 1, .05, 150);
  const direction = new THREE.Vector3(6.8, 5.2, 12.8).normalize();
  let selectedView = 'overview';
  const views = {
    overview: { target:[0, -.1, 0], span:13.8, direction },
    compressor: { target:[-2.65, .12, 0], span:5.8, direction:new THREE.Vector3(1.6, 1.1, 3).normalize() },
    combustors: { target:[1.55, .1, 0], span:4.7, direction:new THREE.Vector3(1.5, 1.5, 3).normalize() },
    turbine: { target:[3.5, .0, 0], span:4.5, direction:new THREE.Vector3(1.6, 1.4, 3).normalize() },
    end: { target:[-1.5, 0, 0], span:4.8, direction:new THREE.Vector3(-1, .04, .025).normalize() },
  };

  function fitView(name) {
    selectedView = name;
    const view = views[name] || views.overview;
    const aspect = Math.max(camera.aspect, .35);
    const verticalSpan = Math.max(view.span / aspect, name === 'overview' ? 4.9 : 3.7);
    const distance = verticalSpan / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)));
    controls.target.fromArray(view.target);
    camera.position.copy(controls.target).addScaledVector(view.direction, distance * 1.07);
    controls.update();
    element.dataset.camera = name;
    element.querySelector('select').value = name;
  }

  function updateUI() {
    const rotating = state.playing && state.rpm !== null && state.rpm > 0;
    element.dataset.rpm = state.rpm === null ? 'unavailable' : String(state.rpm);
    element.dataset.rotating = String(rotating);
    element.dataset.mode = state.mode;
    element.dataset.labels = String(state.labels);
    element.querySelector('[data-role="rpm"]').textContent = state.rpm === null ? 'SOURCE SPEED UNAVAILABLE' : `${state.rpm.toLocaleString('en-US', {maximumFractionDigits:4})} RPM · SOURCE`;
    const rotate = element.querySelector('[data-action="rotate"]');
    rotate.disabled = state.rpm === null || state.rpm <= 0;
    const actionText = rotating ? 'Pause rotor' : 'Rotate rotor';
    rotate.setAttribute('aria-label', actionText);
    element.querySelector('[data-role="play-text"]').textContent = actionText;
    element.querySelector('[data-role="play-icon"]').textContent = rotating ? 'Ⅱ' : '▷';
    element.querySelector('[data-role="motion"]').textContent = state.rpm === null ? 'Rotation unavailable' : state.rpm === 0 ? 'Shaft stopped · 0 RPM' : !state.playing ? 'Inspection paused' : state.rpm > 12000 ? 'Visual speed capped · source above 12,000 RPM' : 'Visual rotation · 1/600 actual speed';
    element.querySelectorAll('[data-mode]').forEach(button => {
      if (button.tagName === 'BUTTON') button.setAttribute('aria-pressed', String(button.dataset.mode === state.mode));
    });
    model.casing.visible = state.mode === 'exterior';
    element.querySelector('[data-role="view-name"]').textContent = state.mode === 'exterior' ? 'ASSEMBLY / COMPLETE CASING' : 'SECTION / UPPER CASING REMOVED';
    const labelsButton = element.querySelector('[data-action="labels"]');
    labelsButton.setAttribute('aria-label', state.labels ? 'Hide labels' : 'Show labels');
    labelsButton.setAttribute('aria-pressed', String(state.labels));
    annotations.hidden = !state.labels;
  }

  function update(data) {
    const value = data?.rpm;
    state.rpm = typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null;
    updateUI();
  }

  function dispose() {
    stopped = true;
    cancelAnimationFrame(animationId);
    abort.abort();
    resizeObserver?.disconnect();
    intersectionObserver?.disconnect();
    controls?.dispose();
    model?.dispose?.();
    // Sets avoid disposing shared instanced geometry/materials repeatedly.
    const geometries = new Set(), materials = new Set();
    scene.traverse(object => {
      if (object.geometry) geometries.add(object.geometry);
      if (object.material) (Array.isArray(object.material) ? object.material : [object.material]).forEach(m => materials.add(m));
    });
    geometries.forEach(item => item.dispose());
    materials.forEach(item => item.dispose());
    environmentTarget?.dispose();
    shadowTexture?.dispose();
    renderer?.dispose();
    renderer?.forceContextLoss();
    renderer?.domElement.remove();
    annotations.replaceChildren();
    delete element.dataset.ready;
  }

  try {
    renderer = new THREE.WebGLRenderer({antialias:true, alpha:true, powerPreference:'high-performance'});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.6));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.setClearColor(0x000000, 0);
    renderer.domElement.setAttribute('aria-label', 'Interactive three-dimensional gas turbine');
    renderer.domElement.setAttribute('role', 'img');
    renderer.domElement.tabIndex = 0;
    viewport.prepend(renderer.domElement);
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = .08;
    controls.minDistance = 2.4;
    controls.maxDistance = 45;
    controls.maxPolarAngle = Math.PI * .86;
    controls.zoomSpeed = .75;
    controls.listenToKeyEvents(renderer.domElement);
    const pmrem = new THREE.PMREMGenerator(renderer);
    const room = new RoomEnvironment();
    environmentTarget = pmrem.fromScene(room, .035);
    scene.environment = environmentTarget.texture;
    scene.environmentIntensity = .85;
    room.dispose();
    pmrem.dispose();
    scene.add(new THREE.HemisphereLight(0xddeaff, 0x28313d, 1.3));
    const keyLight = new THREE.DirectionalLight(0xfff2dc, 3.5);
    keyLight.position.set(-3, 8, 6);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.set(2048, 2048);
    Object.assign(keyLight.shadow.camera, {left:-9,right:9,top:6,bottom:-6,near:.5,far:25});
    keyLight.shadow.normalBias = .025;
    keyLight.shadow.bias = -.0002;
    keyLight.shadow.radius = 3;
    scene.add(keyLight);
    const rim = new THREE.DirectionalLight(0xa9d4ff, 3.0);
    rim.position.set(3, 4, -5);
    scene.add(rim);
    const fill = new THREE.DirectionalLight(0xffffff, .9);
    fill.position.set(-7, 1, 2);
    scene.add(fill);
    model = buildTurbine();
    scene.add(model.root);
    element.dataset.architecture = JSON.stringify(model.stats);

    const ground = new THREE.Mesh(new THREE.PlaneGeometry(40, 25), new THREE.ShadowMaterial({opacity:.27}));
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -1.91;
    ground.receiveShadow = true;
    scene.add(ground);
    const shadowCanvas = document.createElement('canvas');
    shadowCanvas.width = shadowCanvas.height = 128;
    const context = shadowCanvas.getContext('2d');
    const gradient = context.createRadialGradient(64, 64, 5, 64, 64, 64);
    gradient.addColorStop(0, 'rgba(0,0,0,.52)');
    gradient.addColorStop(1, 'rgba(0,0,0,0)');
    context.fillStyle = gradient;
    context.fillRect(0, 0, 128, 128);
    shadowTexture = new THREE.CanvasTexture(shadowCanvas);
    const contact = new THREE.Mesh(new THREE.PlaneGeometry(17, 6), new THREE.MeshBasicMaterial({map:shadowTexture,transparent:true,depthWrite:false}));
    contact.rotation.x = -Math.PI / 2;
    contact.position.y = -1.90;
    scene.add(contact);
    for (const label of model.labels) {
      const node = document.createElement('div');
      node.className = 'model-label';
      node.textContent = label.text;
      if (label.detail) { const detail = document.createElement('small'); detail.textContent = label.detail; node.append(detail); }
      annotations.append(node);
      labelElements.push({node, position:label.position});
    }

    function resize() {
      const width = viewport.clientWidth, height = viewport.clientHeight;
      if (!width || !height) return;
      const oldAspect = camera.aspect;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
      if (Math.abs(oldAspect - camera.aspect) > .25) fitView(selectedView);
    }
    resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(viewport);
    intersectionObserver = new IntersectionObserver(entries => { visible = entries[0].isIntersecting; });
    intersectionObserver.observe(element);
    resize();
    fitView('overview');
    update(initialData);
    const listen = (node, type, handler) => node.addEventListener(type, handler, {signal:abort.signal});
    element.querySelectorAll('button[data-mode]').forEach(button => listen(button, 'click', () => {state.mode = button.dataset.mode; updateUI();}));
    listen(element.querySelector('[data-action="rotate"]'), 'click', () => {state.playing = !state.playing; updateUI();});
    listen(element.querySelector('[data-action="labels"]'), 'click', () => {state.labels = !state.labels; updateUI();});
    listen(element.querySelector('[data-action="reset"]'), 'click', () => fitView('overview'));
    listen(element.querySelector('select'), 'change', event => fitView(event.target.value));
    listen(renderer.domElement, 'webglcontextlost', event => {
      event.preventDefault();
      cancelAnimationFrame(animationId);
      showError('The 3D graphics context was interrupted. Retry to reload the model.');
    });
    let last = performance.now();
    const projected = new THREE.Vector3();
    function animate(now) {
      if (stopped) return;
      animationId = requestAnimationFrame(animate);
      const dt = (now - last) / 1000;
      last = now;
      if (!visible || document.hidden) return;
      if (state.playing) model.rotor.rotation.x = (model.rotor.rotation.x + rotationStep(state.rpm, dt)) % TAU;
      element.dataset.rotorAngle = model.rotor.rotation.x.toFixed(6);
      controls.update();
      if (state.labels) {
        const width = viewport.clientWidth, height = viewport.clientHeight;
        for (const {node,position} of labelElements) {
          projected.copy(position).project(camera);
          const x = (projected.x * .5 + .5) * width, y = (-projected.y * .5 + .5) * height;
          node.hidden = projected.z > 1 || projected.z < -1 || x < 50 || x > width - 50 || y < 60 || y > height - 20;
          node.style.left = `${x}px`;
          node.style.top = `${y - 25}px`;
        }
      }
      renderer.render(scene, camera);
      element.dataset.triangles = String(renderer.info.render.triangles);
    }
    renderer.render(scene, camera);
    element.querySelector('.loading')?.remove();
    element.dataset.rotorAngle = '0.000000';
    element.dataset.ready = 'true';
    animationId = requestAnimationFrame(animate);
  } catch (error) {
    dispose();
    showError('The 3D view could not start. Enable graphics acceleration in your browser, then retry.');
    console.error('Turbine viewer initialization failed:', error);
  }

  function showError(message) {
    element.dataset.ready = 'false';
    element.dataset.rotating = 'false';
    element.querySelector('.loading')?.remove();
    viewport.querySelector('.viewer-error')?.remove();
    const panel = document.createElement('div');
    panel.className = 'viewer-error';
    panel.setAttribute('role', 'alert');
    const text = document.createElement('span');
    text.textContent = message;
    const retry = document.createElement('button');
    retry.type = 'button'; retry.textContent = 'Retry 3D view';
    retry.onclick = () => { dispose(); panel.remove(); const next = createViewer(element, latestData); instance.replacement = next; };
    panel.append(text, retry);
    viewport.append(panel);
  }
  const instance = { update(data) { latestData = data; if (this.replacement) this.replacement.update(data); else if (model && !stopped) update(data); }, dispose() { this.replacement?.dispose(); dispose(); } };
  return instance;
}

export default function mount({parentElement, data}) {
  const element = parentElement.querySelector('[data-testid="turbine-viewer"]');
  if (!element) return;
  let record = instances.get(element);
  if (record) { clearTimeout(record.timer); record.viewer.update(data); }
  else { record = {viewer:createViewer(element, data), timer:null}; instances.set(element, record); }
  return () => {
    record.timer = setTimeout(() => {record.viewer.dispose(); instances.delete(element);}, 0);
  };
}
