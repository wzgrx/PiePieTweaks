import { app } from "../../scripts/app.js";

const RESOLUTIONS = {
    "Flux": {
        "Portrait": ["768x1344", "832x1216", "896x1152", "1024x1536", "1088x1920"],
        "Landscape": ["1344x768", "1216x832", "1152x896", "1536x1024", "1920x1088"],
        "Square": ["1024x1024", "1152x1152"],
    },
    "Wan": {
        "Portrait": ["720x1280", "768x1280", "832x1216", "896x1152", "1024x1792"],
        "Landscape": ["1280x720", "1280x768", "1216x832", "1152x896", "1792x1024"],
        "Square": ["1024x1024", "1280x1280"],
    },
    "Qwen": {
        "Portrait": ["768x1024", "832x1152", "896x1152", "1024x1536", "1152x1728"],
        "Landscape": ["1024x768", "1152x832", "1152x896", "1536x1024", "1728x1152"],
        "Square": ["1024x1024", "1152x1152"],
    },
    "SD1.5": {
        "Portrait": ["512x768", "448x704", "384x640", "512x832", "576x896"],
        "Landscape": ["768x512", "704x448", "640x384", "832x512", "896x576"],
        "Square": ["512x512", "576x576"],
    },
    "SDXL": {
        "Portrait": ["896x1152", "832x1216", "768x1344", "1024x1536", "960x1728"],
        "Landscape": ["1152x896", "1216x832", "1344x768", "1536x1024", "1728x960"],
        "Square": ["1024x1024", "1152x1152"],
    },
    "Pony": {
        "Portrait": ["896x1152", "832x1216", "768x1344", "1024x1536", "960x1728"],
        "Landscape": ["1152x896", "1216x832", "1344x768", "1536x1024", "1728x960"],
        "Square": ["1024x1024", "1152x1152"],
    },
};

app.registerExtension({
    name: "PiePieDesign.ResolutionPicker",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "PiePieResolutionPicker") {
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                onNodeCreated?.apply(this, arguments);
                
                const typeWidget = this.widgets.find(w => w.name === "type");
                const orientationWidget = this.widgets.find(w => w.name === "orientation");
                const resolutionWidget = this.widgets.find(w => w.name === "resolution");
                
                const updateResolutions = () => {
                    const selectedType = typeWidget.value;
                    const selectedOrientation = orientationWidget.value;
                    
                    let resolutions = [];
                    
                    if (selectedType === "CUSTOM") {
                        resolutions = ["Use Custom Width/Height"];
                    } else if (selectedType === "ALL") {
                        if (selectedOrientation === "ALL") {
                            // Show ALL resolutions from ALL model types
                            const allRes = new Set();
                            for (const modelType in RESOLUTIONS) {
                                for (const orientation in RESOLUTIONS[modelType]) {
                                    RESOLUTIONS[modelType][orientation].forEach(res => allRes.add(res));
                                }
                            }
                            resolutions = Array.from(allRes).sort((a, b) => {
                                const aNum = parseInt(a.split('x')[0].split(' ')[0]);
                                const bNum = parseInt(b.split('x')[0].split(' ')[0]);
                                return aNum - bNum;
                            });
                        } else {
                            const allRes = new Set();
                            for (const modelType in RESOLUTIONS) {
                                const orientationRes = RESOLUTIONS[modelType][selectedOrientation] || [];
                                orientationRes.forEach(res => allRes.add(res));
                            }
                            resolutions = Array.from(allRes).sort((a, b) => {
                                const aNum = parseInt(a.split('x')[0].split(' ')[0]);
                                const bNum = parseInt(b.split('x')[0].split(' ')[0]);
                                return aNum - bNum;
                            });
                        }
                    } else {
                        if (selectedOrientation === "ALL") {
                            resolutions = [];
                            for (const orientation in RESOLUTIONS[selectedType]) {
                                resolutions.push(...RESOLUTIONS[selectedType][orientation]);
                            }
                        } else {
                            resolutions = RESOLUTIONS[selectedType]?.[selectedOrientation] || ["1024x1024"];
                        }
                    }
                    
                    resolutionWidget.options.values = resolutions;
                    
                    if (!resolutions.includes(resolutionWidget.value)) {
                        resolutionWidget.value = resolutions[0];
                    }
                    
                    if (resolutionWidget.callback) {
                        resolutionWidget.callback(resolutionWidget.value);
                    }
                };
                
                const originalTypeCallback = typeWidget.callback;
                typeWidget.callback = function(value) {
                    if (originalTypeCallback) {
                        originalTypeCallback.apply(this, arguments);
                    }
                    updateResolutions();
                }.bind(this);
                
                const originalOrientationCallback = orientationWidget.callback;
                orientationWidget.callback = function(value) {
                    if (originalOrientationCallback) {
                        originalOrientationCallback.apply(this, arguments);
                    }
                    updateResolutions();
                }.bind(this);
                
                updateResolutions();
                
                this.setSize(this.computeSize());
            };
        }
    }
});