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

function createTable() {
    let oldTbl = document.getElementById('perftable');
    if (oldTbl)
        oldTbl.remove()

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
    for (let stageName in cpuNodes) {
        for (let threadName in cpuNodes[stageName]) {
            for(let node in cpuNodes[stageName][threadName]) {
                if (minTime == null || node.availableTime < minTime)
                    minTime = node.availableTime;
                if (maxTime == null || maxTime < node.endTime)
                    maxTime = node.endTime;
            }
        }
    }

    for (let stageName in cpuNodes) {
        // console.log(key, yourobject[key]);
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
            threadTr.appendChild(contentTd);

            threadBody.appendChild(threadTr);
            threadTbl.appendChild(threadBody);
            threadsNamesTd.appendChild(threadTbl);
        }
        stageTr.appendChild(threadsNamesTd);

        // let threadContentTd = document.createElement('td');
        // for (let _ in cpuNodes[stageName]) {
        //     let threadTbl = document.createElement('table');
        //     threadTbl.className = 'threadtable';
        //     let threadBody = document.createElement('tbody');
        //     let threadTr = document.createElement('tr');

        //     let contentTd = document.createElement('td');
        //     contentTd.className = 'contenttd';
        //     threadTr.appendChild(contentTd);

        //     threadBody.appendChild(threadTr);
        //     threadTbl.appendChild(threadBody);
        //     threadContentTd.appendChild(threadTbl);
        // }
        // stageTr.appendChild(threadContentTd);

        tbdy.appendChild(stageTr);
    }
    tbl.appendChild(tbdy);
    body.appendChild(tbl);
}
