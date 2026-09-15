import { app } from "../../../scripts/app.js";
import * as THREE from "./lib/three.module.min.js";
import { PanoramaViewer } from "./viewer.js";

app.registerExtension({
    name: "LatLong.PanoViewer",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!["PanoramaViewerNode", "PanoramaVideoViewerNode", "LatLong Compare Panorama"].includes(nodeData.name)) return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            const container = document.createElement("div");
            container.style.cssText = "position:relative;background:#111;overflow:hidden;width:100%;height:300px";
            const widget = this.addDOMWidget("panorama", "preview", container, { serialize: false, hideOnZoom: false });
            widget.computeSize = function (width) {
                const height = Math.max(220, width * .65);
                this.computedHeight = height;
                container.style.height = `${height}px`;
                return [width, height];
            };
            this.viewer = new PanoramaViewer(container, THREE);
            this.setSize([Math.max(420, this.size[0]), Math.max(340, this.size[1])]);
            widget.options.afterResize = () => this.viewer.resize();
            return result;
        };
        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output) {
            const result = executed?.apply(this, arguments);
            const scalar = v => Array.isArray(v) ? v.join("") : v;
            if (output?.pano_video_frames) {
                this.viewer.setFrames(output.pano_video_frames.flat(), Number(scalar(output.fps)), output.compare_frames?.flat() || []);
            } else if (output?.pano_image) {
                this.viewer.setFrames([scalar(output.pano_image)], 1, output.compare_image ? [scalar(output.compare_image)] : []);
            }
            return result;
        };
        const removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            this.viewer?.dispose();
            return removed?.apply(this, arguments);
        };
    }
});
