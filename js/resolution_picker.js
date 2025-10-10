import { app } from "../../scripts/app.js";

let RESOLUTIONS = {};

fetch('extensions/PiePieTweaks/resolutions.json')
    .then(response => {
        console.log('[PiePie] Fetch response status:', response.status);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        return response.json();
    })
    .then(resolutionsData => {
        for (const modelType in resolutionsData) {
            RESOLUTIONS[modelType] = {};
            for (const orientation in resolutionsData[modelType]) {
                RESOLUTIONS[modelType][orientation] = resolutionsData[modelType][orientation].map(
                    (res) => `${res[0]}x${res[1]}`
                );
            }
        }
    })
    .catch(error => {
        console.error('[PiePie] Failed to load resolutions.json:', error);
    });

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
                
                const waitForResolutions = () => {
                    return new Promise((resolve) => {
                        const checkLoaded = () => {
                            if (Object.keys(RESOLUTIONS).length > 0) {
                                resolve();
                            } else {
                                setTimeout(checkLoaded, 50);
                            }
                        };
                        checkLoaded();
                    });
                };
                
                const updateResolutions = async () => {
                    await waitForResolutions();
                    const selectedType = typeWidget.value;
                    const selectedOrientation = orientationWidget.value;
                    
                    let resolutions = [];
                    
                    if (selectedType === "CUSTOM") {
                        resolutions = ["Use Custom Width/Height"];
                    } else if (selectedType === "ALL") {
                        if (selectedOrientation === "ALL") {
                            const allRes = new Set();
                            for (const modelType in RESOLUTIONS) {
                                for (const orientation in RESOLUTIONS[modelType]) {
                                    RESOLUTIONS[modelType][orientation].forEach(res => allRes.add(res));
                                }
                            }
                            resolutions = Array.from(allRes).sort((a, b) => {
                                const aNum = parseInt(a.split('x')[0]);
                                const bNum = parseInt(b.split('x')[0]);
                                return aNum - bNum;
                            });
                        } else {
                            const allRes = new Set();
                            for (const modelType in RESOLUTIONS) {
                                const orientationRes = RESOLUTIONS[modelType][selectedOrientation] || [];
                                orientationRes.forEach(res => allRes.add(res));
                            }
                            resolutions = Array.from(allRes).sort((a, b) => {
                                const aNum = parseInt(a.split('x')[0]);
                                const bNum = parseInt(b.split('x')[0]);
                                return aNum - bNum;
                            });
                        }
                    } else {
                        if (selectedOrientation === "ALL") {
                            resolutions = [];
                            if (RESOLUTIONS[selectedType]) {
                                for (const orientation in RESOLUTIONS[selectedType]) {
                                    resolutions.push(...RESOLUTIONS[selectedType][orientation]);
                                }
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