let captureData = {
var viewMatrix = [[1, 0, 0], [0, 1, 0]];

function copyMat(m) {
    let ret = [[1, 0, 0], [0, 1, 0]];
    for (let row = 0; row < 2; ++row)
        for (let col = 0; col < 3; ++col)
            ret[row][col] = m[row][col];
    return ret;
}

function mulMat(a, b) {
    let ret = [[1, 0, 0], [0, 1, 0]];
    for (let row = 0; row < 2; ++row)
        for (let col = 0; col < 3; ++col)
            ret[row][col] = a[row][0] * b[0][col] + a[row][1] * b[1][col] + (col == 2 ? a[row][2] : 0);
    return ret;
}

function getTranslation(dir) {
    return [[1, 0, dir.x], [0, 1, dir.y]];
}

function getScaling(val) {
    return [[val.x, 0, 0], [0, val.y, 0]];
}

function transformPoint(mat, p) {
    return {
        x: mat[0][0] * p.x + mat[0][1] * p.y + mat[0][2],
        y: mat[1][0] * p.x + mat[1][1] * p.y + mat[1][2]
    };
}

function dot(p1, p2) {
    return p1.x * p2.x + p1.y * p2.y;
}

function invert(mat) {
    let t = {x: -mat[0][2], y: -mat[1][2]};
    // mat = SR * T
    // inv(mat) = inv(SR * T)
    // inv(mat) = inv(T) * inv(SR)
    let invDet = 1.0 / (mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]);
    let invMat = [[invDet * mat[1][1], - invDet * mat[0][1], 0], [- invDet * mat[1][0], invDet * mat[0][0], 0]]
    return mulMat(getTranslation(t), invMat);
}

function diff(to, from) {
    return {
        x: to.x - from.x,
        y: to.y - from.y
    };
}

function translatePoint(p, dir) {
    return {
        x: p.x + dir.x,
        y: p.y + dir.y
    };
}

function invTransformPoint(mat, p) {
    let xCol = {x: mat[0][0], y: mat[1][0]};
    let yCol = {x: mat[0][1], y: mat[1][1]};
    let t = {x: -mat[0][2], y: -mat[1][2]};
    return {
        x: dot(translatePoint(p, t), xCol),
        y: dot(translatePoint(p, t), yCol)
    };
}

function invTransformDir(mat, dir) {
    let xCol = {x: mat[0][0], y: mat[1][0]};
    let yCol = {x: mat[0][1], y: mat[1][1]};
    return {
        x: dot(dir, xCol),
        y: dot(dir, yCol)
    };
}

function transformDir(mat, dir) {
    return {
        x: mat[0][0] * dir.x + mat[0][1] * dir.y,
        y: mat[1][0] * dir.x + mat[1][1] * dir.y
    };
}

var dragStartMouse = null;
var dragStartMat = null;

function onDocumentDragStart(e) {
    dragStartMat = copyMat(viewMatrix);
    // dragStartMouse = transformPoint(viewMatrix, {x: e.clientX, y: e.clientY});
    dragStartMouse = {x: e.clientX, y: e.clientY};
}

function onDocumentDragEnd(e) {
    dragStartMat = null;
    dragStartMouse = null;
}

function applyViewTransform() {
    let table = document.getElementById('perftable');
    if (table) {
        // invView = invert(viewMatrix);
        invView = viewMatrix;
        table.style.transform = `matrix(${invView[0][0]}, ${invView[1][0]}, ${invView[0][1]}, ${invView[1][1]}, ${invView[0][2]}, ${invView[1][2]})`;
    }
}

function onDocumentDrag(e) {
    if (dragStartMat != null) {
        pos = {x: e.clientX, y: e.clientY};
        viewMatrix = mulMat(getTranslation(diff(pos, dragStartMouse)), dragStartMat)
        applyViewTransform();
    }
}
function onDocumentZoom(e) {
    let s = Math.exp(-e.deltaY / 5000.0);
    let scaling = {x: s, y: s};
    viewSpaceZoom = mulMat(
        getTranslation({x: e.clientX, y: e.clientY}),
        mulMat(getScaling(scaling),
        getTranslation({x: -e.clientX, y: -e.clientY}))
    );
    viewMatrix = mulMat(viewSpaceZoom, viewMatrix);
    applyViewTransform();
}

var cpuPackageData = {};

function createTable() {
    let oldTbl = document.getElementById('perftable');
    if (oldTbl)
        oldTbl.remove()

    const timeToWorldUnit = 2;

    let frameIdText = document.getElementById('info_frameid');
    let fpsText = document.getElementById('info_fps');
    let frametimeText = document.getElementById('info_frametime');
    let execDelayText = document.getElementById('info_executiondelay');
    let deviceDelayText = document.getElementById('info_devicedelay');

    frameIdText.innerHTML = captureData.frameId;
    fpsText.innerHTML = Math.round(captureData.fps);
    frametimeText.innerHTML = `${Math.round(captureData.frameTime)}ms`;
    execDelayText.innerHTML = `${Math.round(captureData.executionDelay)}ms`;
    deviceDelayText.innerHTML = `${Math.round(captureData.deviceDelay)}ms`;

    document.onmousedown = onDocumentDragStart;
    document.onmouseup = onDocumentDragEnd;
    document.onmousemove = onDocumentDrag;
    document.onwheel = onDocumentZoom;

    let body = document.getElementsByTagName('body')[0];
    let tbl = document.createElement('table');
    tbl.id = 'perftable';
    let tbdy = document.createElement('tbody');

    cpuNodes = captureData.stageToThreadToPackageList;
    let minTime = null;
    let maxTime = null;
    let minTargetTime = null;
    for (let stageName in cpuNodes) {
        for (let threadName in cpuNodes[stageName]) {
            for(let i in cpuNodes[stageName][threadName]) {
                let node = cpuNodes[stageName][threadName][i];
                if (minTime == null || node.availableTime < minTime)
                    minTime = node.availableTime;
                if (node.frameId == captureData.frameId && (minTargetTime == null || node.availableTime < minTargetTime))
                    minTargetTime = node.availableTime;
                if (maxTime == null || maxTime < node.endTime)
                    maxTime = node.endTime;
            }
        }
    }

    cpuPackageData = {};

    for (let stageName in cpuNodes) {
        let stageTr = document.createElement('tr');

        let stageNameTd = document.createElement('td');
        let stageText = document.createElement('div');
        stageText.innerHTML = stageName
        stageText.className = 'stageName';
        stageNameTd.appendChild(stageText);
        stageNameTd.className = 'nametd';
        stageTr.appendChild(stageNameTd);

        let threadsNamesTd = document.createElement('td');
        for (let threadName in cpuNodes[stageName]) {
            let threadTbl = document.createElement('table');
            threadTbl.className = 'threadtable';
            let threadBody = document.createElement('tbody');
            let threadTr = document.createElement('tr');

            let threadNameTd = document.createElement('td');
            threadNameTd.className = 'nametd';
            let threadNameDiv = document.createElement('div');
            threadNameDiv.innerHTML = threadName;
            threadNameDiv.className = 'threadName';
            threadNameTd.appendChild(threadNameDiv);
            threadTr.appendChild(threadNameTd);

            let contentTd = document.createElement('td');
            contentTd.className = 'contenttd';
            contentTd.style.width = `${timeToWorldUnit * (maxTime - minTime)}px`;

            for(let i in cpuNodes[stageName][threadName]) {
                let node = cpuNodes[stageName][threadName][i];
                let resWaitTime = Math.max(node.startTime - node.resAvailableTime, 0);
                let waitTime = Math.max(node.startTime - node.availableTime, 0);
                if (resWaitTime > 0.05) {
                    waitTime = Math.max(node.resAvailableTime - node.availableTime, 0)
                }
                else
                    resWaitTime = 0;
                let workTime = Math.max(node.endTime - node.startTime, 0);

                let nodeWrapperElem = document.createElement('div');
                nodeWrapperElem.className = "nodeWrapper " + (node.frameId == captureData.frameId ? "currentFrame" : "otherFrame");
                nodeWrapperElem.style.left = `${timeToWorldUnit * (node.availableTime - minTime)}px`
                let nodeElem = document.createElement('div');
                nodeElem.className = "cpuNode";
                cpuPackageData[node.packageId] = {w: nodeWrapperElem, n: node};

                let textWrapper = document.createElement('div');
                textWrapper.className = "textWrapper";
                textWrapper.style.width = `${Math.max(timeToWorldUnit * (waitTime + workTime), 1)}px`;

                let nodeNameElem = document.createElement('div');
                nodeNameElem.className = "nodeText";
                nodeNameElem.innerHTML = `${node.name} (${node.frameId - captureData.frameId})`;
                textWrapper.appendChild(nodeNameElem);

                let nodeTimingElem = document.createElement('div');
                nodeTimingElem.className = "nodeText";
                nodeTimingElem.innerHTML = `${Math.round(node.availableTime - minTargetTime)} | ${Math.round(node.startTime - minTargetTime)} | ${Math.round(node.endTime - minTargetTime)} ms`;
                textWrapper.appendChild(nodeTimingElem);
                nodeElem.appendChild(textWrapper);


                if (resWaitTime > 0) {
                    let resElem = document.createElement('div');
                    resElem.className = "resourceWaitTime";
                    resElem.style.width = `${Math.max(timeToWorldUnit * resWaitTime, 1)}px`;
                    nodeElem.appendChild(resElem);
                }
                if (waitTime > 0.05) {
                    let waitElem = document.createElement('div');
                    waitElem.className = "waitTime";
                    waitElem.style.width = `${Math.max(timeToWorldUnit * waitTime, 1)}px`;
                    nodeElem.appendChild(waitElem);
                }
                let workElem = document.createElement('div');
                workElem.className = "workTime";
                workElem.style.width = `${Math.max(timeToWorldUnit * workTime, 1)}px`;
                nodeElem.appendChild(workElem);


                nodeWrapperElem.appendChild(nodeElem);
                contentTd.appendChild(nodeWrapperElem);
            }
            threadTr.appendChild(contentTd);

            threadBody.appendChild(threadTr);
            threadTbl.appendChild(threadBody);
            threadsNamesTd.appendChild(threadTbl);
        }
        stageTr.appendChild(threadsNamesTd);
        tbdy.appendChild(stageTr);
    }
    tbl.appendChild(tbdy);
    body.appendChild(tbl);

    for (let pkgId in cpuPackageData) {
        // cpuPackageData[node.packageId] = {w: nodeWrapperElem, n: node};
        let info = cpuPackageData[pkgId];
        info.w.onmouseenter = (_) => {
            for (let depended in info.n.depended) {
                let dep = cpuPackageData[info.n.depended[depended]];
                if (dep.n.endTime < info.n.availableTime)
                    dep.w.classList.add("depended");
                else
                    dep.w.classList.add("activeDepended");
            }
            for (let dependent in info.n.dependent) {
                let dep = cpuPackageData[info.n.dependent[dependent]];
                if (dep.n.availableTime < info.n.endTime)
                    dep.w.classList.add("activeDependent");
                else
                    dep.w.classList.add("dependent");
            }
        }
        info.w.onmouseleave = (_) => {
            for (let depended in info.n.depended) {
                let dep = cpuPackageData[info.n.depended[depended]];
                if (dep.n.endTime < info.n.availableTime)
                    dep.w.classList.remove("depended");
                else
                    dep.w.classList.remove("activeDepended");
            }
            for (let dependent in info.n.dependent) {
                let dep = cpuPackageData[info.n.dependent[dependent]];
                if (dep.n.availableTime < info.n.endTime)
                    dep.w.classList.remove("activeDependent");
                else
                    dep.w.classList.remove("dependent");
            }
        }
    }
}
