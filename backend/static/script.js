document.getElementById('singleForm').onsubmit = async (e) => {
    e.preventDefault();
    let form = e.target;
    let data = new FormData(form);
    let res = await fetch('/scan_single', { method: 'POST', body: data });
    let result = await res.json();
    document.getElementById('results').innerText = JSON.stringify(result, null, 2);
};

document.getElementById('multiForm').onsubmit = async (e) => {
    e.preventDefault();
    let form = e.target;
    let data = new FormData(form);
    let res = await fetch('/scan_multiple', { method: 'POST', body: data });
    let result = await res.json();
    document.getElementById('results').innerText = JSON.stringify(result, null, 2);
};

document.getElementById('cameraForm').onsubmit = async (e) => {
    e.preventDefault();
    let form = e.target;
    let data = new URLSearchParams(new FormData(form));
    let res = await fetch('/scan_ip_camera', { method: 'POST', body: data });
    let result = await res.json();
    document.getElementById('results').innerText = JSON.stringify(result, null, 2);
};
