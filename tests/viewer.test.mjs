import test from 'node:test';
import assert from 'node:assert/strict';
import { PanoramaViewer } from '../js/viewer.js';

function fixture() {
    const pending = [], raf = new Map(); let next = 0, disposed = 0;
    const element = () => ({ style: {}, clientWidth: 400, clientHeight: 240, appendChild() {}, append() {}, addEventListener() {}, removeEventListener() {}, setAttribute() {}, setPointerCapture() {} });
    globalThis.document = { ...element(), hidden: false, createElement: element };
    globalThis.window = { devicePixelRatio: 1 };
    globalThis.IntersectionObserver = class { observe() {} disconnect() {} };
    globalThis.ResizeObserver = class { observe() {} disconnect() {} };
    globalThis.requestAnimationFrame = callback => { raf.set(++next, callback); return next; };
    globalThis.cancelAnimationFrame = id => raf.delete(id);
    const THREE = {
        Scene: class {}, PerspectiveCamera: class { lookAt() {} updateProjectionMatrix() {} },
        WebGLRenderer: class { constructor() { this.domElement = element(); } setPixelRatio() {} setSize() {} setScissorTest() {} setScissor() {} render() {} dispose() { disposed++; } forceContextLoss() {} },
        TextureLoader: class { load(url, ok, progress, fail) { pending.push({ url, ok, fail }); } }
    };
    const texture = id => ({ id, dispose() { disposed++; } });
    return { viewer: new PanoramaViewer(element(), THREE), pending, raf, texture, get disposed() { return disposed; } };
}
const flush = () => new Promise(resolve => setImmediate(resolve));

test('new execution discards late textures and keeps one clock', async () => {
    const f = fixture(); f.viewer.setFrames(['old','b'],30); f.viewer.setFrames(['new','c'],30);
    assert.equal(f.raf.size,1);
    f.pending[1].ok(f.texture('new')); await flush();
    f.pending[0].ok(f.texture('old')); await flush();
    assert.equal(f.viewer.scene.background.id,'new'); assert.equal(f.disposed,1);
    f.viewer.dispose(); assert.equal(f.raf.size,0);
});
test('loading stays locked until decode finishes; seeking queues latest frame', async () => {
    const f = fixture(); f.viewer.setFrames(['a','b','c'],30); f.viewer.showFrame(1); f.viewer.showFrame(2);
    assert.equal(f.pending.length,1); assert.equal(f.viewer.loading,true);
    f.pending[0].ok(f.texture('a')); await flush();
    assert.equal(f.pending[1].url,'c'); f.pending[1].ok(f.texture('c')); await flush();
    assert.equal(f.viewer.frame,2); assert.equal(f.viewer.loading,false); assert.equal(f.disposed,1); f.viewer.dispose();
});
test('dispose releases current and late arriving textures', async () => {
    const f = fixture(); f.viewer.setFrames(['a'],1); f.viewer.dispose();
    f.pending[0].ok(f.texture('a')); await flush(); assert.equal(f.disposed,2);
});
test('comparison waits for both images and cleans partial failures', async () => {
    const f = fixture(); f.viewer.setFrames(['a'],1,['b']);
    f.pending[0].ok(f.texture('a')); f.pending[1].fail(new Error('decode')); await flush();
    assert.equal(f.disposed,1); assert.equal(f.viewer.playing,false); assert.equal(f.viewer.loading,false); f.viewer.dispose();
});
test('single frame timeline uses exact frame indices', async () => {
    const f = fixture(); f.viewer.setFrames(['a'],1); f.pending[0].ok(f.texture('a')); await flush();
    assert.equal(f.viewer.slider.max,0); assert.equal(f.viewer.slider.value,0); assert.equal(f.viewer.counter.textContent,'1/1'); f.viewer.dispose();
});
