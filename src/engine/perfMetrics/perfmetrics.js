let captureData = {
}

function createTable() {
    let oldTbl = document.getElementById('perftable');
    if (oldTbl)
        oldTbl.remove()

    let body = document.getElementsByTagName('body')[0];
    let tbl = document.createElement('table');
    tbl.id = 'perftable';
    let tbdy = document.createElement('tbody');

    cpuNodes = captureData.stageToThreadToPackageList;

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