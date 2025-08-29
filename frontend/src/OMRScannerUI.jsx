import React, { useState } from "react";
import axios from "axios";

export default function OMRScannerUI() {
  const [answerKeyFile, setAnswerKeyFile] = useState(null);
  const [omrFile, setOmrFile] = useState(null);
  const [multipleFiles, setMultipleFiles] = useState([]);
  const [ipCamUrl, setIpCamUrl] = useState("http://10.150.114.196:8080/shot.jpg");
  const [studentName, setStudentName] = useState("Unknown");
  const [results, setResults] = useState("");

  // ----------------------------
  // Upload Answer Key
  // ----------------------------
  const handleUploadAnswerKey = async () => {
    if (!answerKeyFile) return alert("Please select an answer key file");
    const formData = new FormData();
    formData.append("file", answerKeyFile);

    try {
      const res = await axios.post("http://127.0.0.1:5000/load-answer-key", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      alert(res.data.message);
    } catch (err) {
      alert(err.response?.data?.error || err.message);
    }
  };

  // ----------------------------
  // Scan Single Sheet
  // ----------------------------
  const handleScanSingle = async () => {
    if (!omrFile) return alert("Please select an OMR sheet file");
    const formData = new FormData();
    formData.append("file", omrFile);

    try {
      const res = await axios.post("http://127.0.0.1:5000/scan-single", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResults(JSON.stringify(res.data, null, 2));
    } catch (err) {
      alert(err.response?.data?.error || err.message);
    }
  };

  // ----------------------------
  // Scan Multiple Sheets
  // ----------------------------
  const handleScanMultiple = async () => {
    if (multipleFiles.length === 0) return alert("Please select multiple files");
    const formData = new FormData();
    multipleFiles.forEach((f) => formData.append("files", f));

    try {
      const res = await axios.post("http://127.0.0.1:5000/scan-multiple", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResults(JSON.stringify(res.data, null, 2));
    } catch (err) {
      alert(err.response?.data?.error || err.message);
    }
  };

  // ----------------------------
  // Scan from IP Camera
  // ----------------------------
  const handleScanCamera = async () => {
    if (!ipCamUrl) return alert("Please enter IP camera URL");
    try {
      const res = await axios.post("http://127.0.0.1:5000/scan-camera", { url: ipCamUrl });
      setResults(JSON.stringify(res.data, null, 2));
    } catch (err) {
      alert(err.response?.data?.error || err.message);
    }
  };

  return (
    <div style={{ padding: "20px" }}>
      <h2>OMR Scanner - React UI</h2>

      <div>
        <label>Answer Key: </label>
        <input type="file" onChange={(e) => setAnswerKeyFile(e.target.files[0])} />
        <button onClick={handleUploadAnswerKey}>Load Answer Key</button>
      </div>

      <div style={{ marginTop: "10px" }}>
        <label>Student Name: </label>
        <input value={studentName} onChange={(e) => setStudentName(e.target.value)} />
      </div>

      <div style={{ marginTop: "10px" }}>
        <label>OMR Sheet: </label>
        <input type="file" onChange={(e) => setOmrFile(e.target.files[0])} />
        <button onClick={handleScanSingle}>Scan Single</button>
      </div>

      <div style={{ marginTop: "10px" }}>
        <label>Multiple OMR Sheets: </label>
        <input type="file" multiple onChange={(e) => setMultipleFiles([...e.target.files])} />
        <button onClick={handleScanMultiple}>Scan Multiple</button>
      </div>

      <div style={{ marginTop: "10px" }}>
        <label>IP Camera URL: </label>
        <input value={ipCamUrl} onChange={(e) => setIpCamUrl(e.target.value)} />
        <button onClick={handleScanCamera}>Scan from Camera</button>
      </div>

      <div style={{ marginTop: "20px" }}>
        <h3>Results:</h3>
        <textarea
          rows={20}
          cols={100}
          value={results}
          readOnly
          style={{ fontFamily: "monospace" }}
        />
      </div>
    </div>
  );
}
