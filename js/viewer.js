// One camera and clock for playback and comparison. Only the displayed textures are retained.
export class PanoramaViewer {
    constructor(container, THREE) {
        this.container = container;
        this.THREE = THREE;
        this.scene = new THREE.Scene();
        this.comparison = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, 1, .1, 10);
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        this.renderer.domElement.style.cssText = "display:block;width:100%;height:100%;position:absolute;inset:0";
        container.appendChild(this.renderer.domElement);
        this.lon = 0; this.lat = 0; this.frame = 0; this.frames = []; this.otherFrames = [];
        this.fps = 30; this.playing = false; this.visible = true; this.disposed = false;
        this.generation = 0; this.request = 0; this.raf = null; this.lastTime = 0;
        this.loading = false; this.pendingFrame = null; this.wipe = .5;
        this.events = [];
        const controls = document.createElement("div");
        controls.style.cssText = "position:absolute;bottom:8px;left:8px;right:8px;display:flex;flex-wrap:wrap;gap:8px;align-items:center;background:#161616dd;padding:8px;color:white";
        this.button = document.createElement("button"); this.button.textContent = "Play";
        this.slider = document.createElement("input"); this.slider.type = "range"; this.slider.min = 0; this.slider.step = 1; this.slider.style.cssText = "flex:1;min-width:40px;width:80px"; this.slider.setAttribute("aria-label", "Frame");
        this.counter = document.createElement("span");
        this.compareSlider = document.createElement("input"); this.compareSlider.type = "range"; this.compareSlider.min = 0; this.compareSlider.max = 1; this.compareSlider.step = .01; this.compareSlider.value = .5; this.compareSlider.setAttribute("aria-label", "Comparison wipe"); this.compareSlider.hidden = true;
        this.compareSlider.style.cssText = "flex:1;min-width:40px;width:80px"; this.compareSlider.title = "Before / after wipe";
        controls.append(this.button, this.slider, this.counter, this.compareSlider); container.appendChild(controls);
        this.on(this.button, "click", () => { this.playing = !this.playing; this.lastTime = performance.now(); this.updateControls(); this.schedule(); });
        this.on(this.slider, "input", () => { this.playing = false; this.showFrame(Number(this.slider.value)); });
        this.on(this.compareSlider, "input", () => { this.wipe = Number(this.compareSlider.value); this.draw(); });
        const canvas = this.renderer.domElement;
        this.on(canvas, "pointerdown", e => { e.preventDefault(); e.stopPropagation(); this.pointer = [e.clientX, e.clientY, this.lon, this.lat]; canvas.setPointerCapture(e.pointerId); });
        this.on(canvas, "pointermove", e => { if (!this.pointer) return; this.lon = this.pointer[2] + (this.pointer[0] - e.clientX) * .15; this.lat = Math.max(-89.9, Math.min(89.9, this.pointer[3] + (e.clientY - this.pointer[1]) * .15)); this.draw(); });
        this.on(canvas, "pointerup", () => { this.pointer = null; });
        this.on(canvas, "pointercancel", () => { this.pointer = null; });
        this.on(canvas, "wheel", e => { e.preventDefault(); e.stopPropagation(); this.camera.fov = Math.max(20, Math.min(120, this.camera.fov + e.deltaY * .05)); this.camera.updateProjectionMatrix(); this.draw(); }, { passive: false });
        this.on(document, "visibilitychange", () => { this.lastTime = performance.now(); this.schedule(); });
        this.observer = new IntersectionObserver(entries => { this.visible = entries[0].isIntersecting; this.lastTime = performance.now(); this.schedule(); });
        this.observer.observe(container);
        this.resizeObserver = new ResizeObserver(() => this.resize()); this.resizeObserver.observe(container);
        this.resize();
    }
    on(target, name, callback, options) { target.addEventListener(name, callback, options); this.events.push([target, name, callback, options]); }
    resize() {
        if (this.disposed) return;
        const w = Math.max(1, this.container.clientWidth), h = Math.max(1, this.container.clientHeight);
        this.camera.aspect = w / h; this.camera.updateProjectionMatrix(); this.renderer.setSize(w, h); this.draw();
    }
    setFrames(frames, fps = 30, otherFrames = []) {
        this.generation++; this.request++; this.loading = false; this.pendingFrame = null;
        this.frames = frames; this.otherFrames = otherFrames; this.fps = Math.max(1, fps); this.frame = 0;
        this.playing = frames.length > 1; this.lastTime = performance.now();
        if (this.raf !== null) cancelAnimationFrame(this.raf);
        this.raf = null;
        for (const scene of [this.scene, this.comparison]) { scene.background?.dispose(); scene.background = null; }
        this.compareSlider.hidden = !otherFrames.length;
        this.showFrame(0); this.schedule();
    }
    load(url) {
        return new Promise((resolve, reject) => new this.THREE.TextureLoader().load(url, resolve, undefined, reject));
    }
    async showFrame(index) {
        if (this.disposed || !this.frames.length) return;
        if (this.loading) { this.pendingFrame = index; return; }
        this.loading = true;
        const generation = this.generation, request = ++this.request;
        const urls = [this.frames[index]];
        if (this.otherFrames.length) urls.push(this.otherFrames[index % this.otherFrames.length]);
        // allSettled lets us release a successful half of a failed comparison load.
        const results = await Promise.allSettled(urls.map(url => this.load(url)));
        const textures = results.filter(r => r.status === "fulfilled").map(r => r.value);
        if (this.disposed || generation !== this.generation || request !== this.request) { textures.forEach(t => t.dispose()); return; }
        this.loading = false;
        if (results.some(r => r.status === "rejected")) {
            textures.forEach(t => t.dispose()); this.playing = false; this.counter.textContent = "Frame could not load";
        } else {
            textures.forEach((texture, i) => { texture.colorSpace = this.THREE.SRGBColorSpace; texture.mapping = this.THREE.EquirectangularReflectionMapping; const scene = i ? this.comparison : this.scene; scene.background?.dispose(); scene.background = texture; });
            this.frame = index; this.updateControls(); this.draw();
        }
        if (this.pendingFrame !== null) { const next = this.pendingFrame; this.pendingFrame = null; this.showFrame(next); }
    }
    updateControls() {
        this.button.textContent = this.playing ? "Pause" : "Play";
        this.button.disabled = this.frames.length < 2; this.slider.max = Math.max(0, this.frames.length - 1); this.slider.value = this.frame;
        this.counter.textContent = `${this.frame + 1}/${this.frames.length}`;
    }
    schedule() {
        if (this.disposed || !this.playing || !this.visible || document.hidden || this.raf !== null) return;
        this.raf = requestAnimationFrame(now => {
            this.raf = null;
            if (now - this.lastTime >= 1000 / this.fps && !this.loading) { this.lastTime = now; this.showFrame((this.frame + 1) % this.frames.length); }
            this.schedule();
        });
    }
    draw() {
        if (this.disposed || !this.visible || document.hidden) return;
        const lon = this.lon * Math.PI / 180, lat = this.lat * Math.PI / 180;
        this.camera.lookAt(Math.cos(lat) * Math.cos(lon), Math.sin(lat), Math.cos(lat) * Math.sin(lon));
        this.renderer.setScissorTest(false); this.renderer.render(this.scene, this.camera);
        if (this.comparison.background) {
            const w = this.container.clientWidth, h = this.container.clientHeight;
            this.renderer.setScissorTest(true); this.renderer.setScissor(Math.round(w * this.wipe), 0, Math.ceil(w * (1 - this.wipe)), h);
            this.renderer.render(this.comparison, this.camera); this.renderer.setScissorTest(false);
        }
    }
    dispose() {
        if (this.disposed) return;
        this.disposed = true; this.generation++; this.playing = false;
        if (this.raf !== null) cancelAnimationFrame(this.raf);
        this.observer.disconnect(); this.resizeObserver.disconnect();
        for (const [target, name, callback, options] of this.events) target.removeEventListener(name, callback, options);
        this.scene.background?.dispose(); this.comparison.background?.dispose(); this.renderer.dispose();
        this.renderer.forceContextLoss(); this.frames = []; this.otherFrames = []; this.events = [];
    }
}
