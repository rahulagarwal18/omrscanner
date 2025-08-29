import { motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { 
  Upload, 
  Camera, 
  FileSpreadsheet, 
  PlayCircle, 
  Download,
  ScanLine,
  Users,
  FileText,
  User,
  IdCard,
  GraduationCap,
  Eye,
  Trash2,
  Settings,
  CheckCircle,
  XCircle,
  AlertCircle,
  RefreshCw,
  X
} from "lucide-react";

// Simple Card components with better spacing
const Card = ({ children, className = "" }) => (
  <div className={`bg-white/10 backdrop-blur-lg rounded-xl shadow-xl border border-white/20 ${className}`}>
    {children}
  </div>
);

const CardContent = ({ children, className = "" }) => (
  <div className={`p-6 ${className}`}>
    {children}
  </div>
);

export default function App() {
  const cursorRef = useRef(null);
  const fileInputRef = useRef(null);
  const multipleFileInputRef = useRef(null);
  const answerKeyRef = useRef(null);
  
  // States with proper initialization
  const [omrFile, setOmrFile] = useState(null);
  const [answerKeyFile, setAnswerKeyFile] = useState(null);
  const [results, setResults] = useState([]);
  const [currentResult, setCurrentResult] = useState(null);
  const [answerKey, setAnswerKey] = useState(null);
  const [ipCamUrl, setIpCamUrl] = useState("http://192.168.1.100:8080/shot.jpg");
  const [loading, setLoading] = useState(false);
  const [answerKeyLoaded, setAnswerKeyLoaded] = useState(false);
  
  // Student details
  const [studentName, setStudentName] = useState("");
  const [regNo, setRegNo] = useState("");
  const [studentClass, setStudentClass] = useState("");
  
  // Export states
  const [selectedFormat, setSelectedFormat] = useState("pdf");
  const [exportLoading, setExportLoading] = useState(false);
  
  // UI States
  const [showResults, setShowResults] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [notification, setNotification] = useState(null);

  const API_BASE_URL = "http://localhost:5000/api";

  // Notification system
  const showNotification = (message, type = "info") => {
    setNotification({ message, type });
    setTimeout(() => setNotification(null), 5000);
  };

  // Custom cursor effect
  useEffect(() => {
    const cursor = cursorRef.current;
    if (!cursor) return;
    
    const moveCursor = (e) => {
      cursor.style.left = `${e.clientX}px`;
      cursor.style.top = `${e.clientY}px`;
    };
    window.addEventListener("mousemove", moveCursor);
    return () => window.removeEventListener("mousemove", moveCursor);
  }, []);

  // Load all results and answer key on component mount
  useEffect(() => {
    fetchAllResults();
    checkAnswerKey();
  }, []);

  const checkAnswerKey = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/get-answer-key`);
      if (response.ok) {
        const data = await response.json();
        if (data.answer_key && Object.keys(data.answer_key).length > 0) {
          setAnswerKey(data.answer_key);
          setAnswerKeyLoaded(true);
        }
      }
    } catch (err) {
      console.error("Failed to check answer key:", err);
    }
  };

const fetchAllResults = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/results`);
      if (response.ok) {
        const data = await response.json();
        console.log("Fetched results:", data);
        setResults(Array.isArray(data?.results) ? data.results : []);
      } else {
        console.error("Failed to fetch results:", response.status, response.statusText);
        setResults([]);
        if (response.status === 500) {
          showNotification("Server error while fetching results. Please check if the backend server is running.", "error");
        }
      }
    } catch (err) {
      console.error("Failed to fetch results:", err);
      setResults([]);
      showNotification("Cannot connect to server. Please ensure the backend is running on localhost:5000", "error");
    }
  };

  // Scan Answer Key from Camera
  const handleScanAnswerKeyFromCamera = async () => {
    if (!ipCamUrl.trim()) {
      showNotification("Please enter IP camera URL", "error");
      return;
    }
    
    setLoading(true);
    
    try {
      const response = await fetch(`${API_BASE_URL}/scan-answer-key`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url: ipCamUrl })
      });
      
      if (response.ok) {
        const data = await response.json();
        setAnswerKey(data.answer_key);
        setAnswerKeyLoaded(true);
        showNotification(data.message, "success");
        console.log("Answer key scanned from camera:", data.answer_key);
      } else {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to scan answer key');
      }
      
    } catch (err) {
      console.error("Answer key scan error:", err);
      showNotification(err.message, "error");
    } finally {
      setLoading(false);
    }
  };

  // Upload Answer Key from File
  const handleUploadAnswerKey = () => answerKeyRef.current?.click();
  
  const handleAnswerKeyChange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    setAnswerKeyFile(file);
    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);
    
    try {
      const response = await fetch(`${API_BASE_URL}/load-answer-key`, {
        method: 'POST',
        body: formData
      });
      
      if (response.ok) {
        const data = await response.json();
        setAnswerKey(data.answer_key);
        setAnswerKeyLoaded(true);
        showNotification(data.message, "success");
        console.log("Answer key loaded:", data.answer_key);
      } else {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to upload answer key');
      }
      
    } catch (err) {
      console.error("Answer key upload error:", err);
      showNotification(err.message, "error");
    } finally {
      setLoading(false);
    }
  };

  // Upload Single OMR Sheet
  const handleUploadOMR = () => fileInputRef.current?.click();
  
  const handleOMRFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    if (!answerKeyLoaded) {
      showNotification("Please load or scan an answer key first!", "error");
      return;
    }
    
    setOmrFile(file);
    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("student_name", studentName || "Unknown Student");
    formData.append("reg_no", regNo || "N/A");
    formData.append("class", studentClass || "N/A");
    
    try {
      const response = await fetch(`${API_BASE_URL}/scan-single`, {
        method: 'POST',
        body: formData
      });
      
      if (response.ok) {
        const data = await response.json();
        setCurrentResult(data);
        await fetchAllResults();
        showNotification("OMR sheet scanned successfully!", "success");
        console.log("Scan result:", data);
      } else {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to scan OMR');
      }
      
    } catch (err) {
      console.error("OMR scan error:", err);
      showNotification(err.message, "error");
    } finally {
      setLoading(false);
    }
  };

  // Upload Multiple OMR Sheets
  const handleUploadMultipleOMR = () => multipleFileInputRef.current?.click();
  
  const handleMultipleOMRFileChange = async (e) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    
    if (!answerKeyLoaded) {
      showNotification("Please load or scan an answer key first!", "error");
      return;
    }
    
    setLoading(true);

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append("files", files[i]);
    }
    
    try {
      const response = await fetch(`${API_BASE_URL}/scan-multiple`, {
        method: 'POST',
        body: formData
      });
      
      if (response.ok) {
        const data = await response.json();
        await fetchAllResults();
        showNotification(`Successfully processed ${data.total_processed} OMR sheets!`, "success");
        console.log("Batch scan results:", data);
      } else {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to scan multiple OMRs');
      }
      
    } catch (err) {
      console.error("Multiple OMR scan error:", err);
      showNotification(err.message, "error");
    } finally {
      setLoading(false);
    }
  };

  // Scan from Camera
  const handleCameraScan = async () => {
    if (!ipCamUrl.trim()) {
      showNotification("Please enter IP camera URL", "error");
      return;
    }
    
    if (!answerKeyLoaded) {
      showNotification("Please load or scan an answer key first!", "error");
      return;
    }
    
    setLoading(true);
    
    try {
      const response = await fetch(`${API_BASE_URL}/scan-camera`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          url: ipCamUrl,
          student_name: studentName || "Unknown Student",
          reg_no: regNo || "N/A",
          class: studentClass || "N/A"
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        setCurrentResult(data);
        await fetchAllResults();
        showNotification("Camera scan completed successfully!", "success");
        console.log("Camera scan result:", data);
      } else {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to scan from camera');
      }
      
    } catch (err) {
      console.error("Camera scan error:", err);
      showNotification(err.message, "error");
    } finally {
      setLoading(false);
    }
  };

  // Fixed Export Results function
  const handleExportResult = async (resultIndex) => {
    if (!results || resultIndex >= results.length || resultIndex < 0) {
      showNotification("Invalid result selection", "error");
      return;
    }
    
    setExportLoading(true);
    
    try {
      // Use the actual result ID instead of index for the API call
      const result = results[resultIndex];
      const resultId = result.id || resultIndex;
      
      console.log("Attempting to export result:", resultId, "Format:", selectedFormat);
      
      // Method 1: Direct download approach
      const response = await fetch(`${API_BASE_URL}/export/${selectedFormat}/${resultId}`);
      
      if (!response.ok) {
        console.error("Export failed:", response.status, response.statusText);
        const errorData = await response.text();
        console.error("Error response:", errorData);
        throw new Error(`Export failed: ${response.statusText}`);
      }

      const data = await response.json();
      console.log("Export response:", data);

      if (data.success && data.download_url) {
        // Download the file using the provided URL
        const downloadResponse = await fetch(`${API_BASE_URL.replace('/api', '')}${data.download_url}`);
        
        if (!downloadResponse.ok) {
          throw new Error(`Download failed: ${downloadResponse.statusText}`);
        }

        const blob = await downloadResponse.blob();
        
        if (blob.size === 0) {
          throw new Error('Downloaded file is empty');
        }

        // Create download link
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = data.filename;
        
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(url);
        
        showNotification(`File exported successfully as ${selectedFormat.toUpperCase()}!`, "success");
      } else {
        throw new Error(data.error || 'Export response invalid');
      }
      
    } catch (err) {
      console.error("Export error:", err);
      showNotification(`Export failed: ${err.message}`, "error");
    } finally {
      setExportLoading(false);
    }
  };

  // Delete result
  const handleDeleteResult = async (resultIndex) => {
    if (!confirm("Are you sure you want to delete this result?")) return;
    
    try {
      const result = results[resultIndex];
      const resultId = result.id || resultIndex;
      
      const response = await fetch(`${API_BASE_URL}/results/${resultId}`, {
        method: 'DELETE'
      });
      
      if (response.ok) {
        await fetchAllResults();
        if (currentResult && results[resultIndex] === currentResult) {
          setCurrentResult(null);
        }
        showNotification("Result deleted successfully!", "success");
      } else {
        throw new Error('Failed to delete result');
      }
    } catch (err) {
      console.error("Delete error:", err);
      showNotification(err.message, "error");
    }
  };

  // Clear all results
  const handleClearAllResults = async () => {
    if (!confirm("Are you sure you want to delete ALL results? This cannot be undone.")) return;
    
    try {
      const response = await fetch(`${API_BASE_URL}/results`, {
        method: 'DELETE'
      });
      
      if (response.ok) {
        setResults([]);
        setCurrentResult(null);
        showNotification("All results cleared successfully!", "success");
      } else {
        throw new Error('Failed to clear results');
      }
    } catch (err) {
      console.error("Clear all error:", err);
      showNotification(err.message, "error");
    }
  };

  return (
    <div className="relative min-h-screen w-full bg-gradient-to-br from-black via-purple-900 to-black overflow-x-hidden text-white">
      {/* Hidden file inputs */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleOMRFileChange}
        accept="image/*"
        style={{ display: "none" }}
      />
      <input
        type="file"
        ref={multipleFileInputRef}
        onChange={handleMultipleOMRFileChange}
        accept="image/*"
        multiple
        style={{ display: "none" }}
      />
      <input
        type="file"
        ref={answerKeyRef}
        onChange={handleAnswerKeyChange}
        accept=".json,.csv"
        style={{ display: "none" }}
      />

      {/* Notification */}
      {notification && (
        <motion.div
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -50 }}
          className="fixed top-4 right-4 z-50 max-w-md"
        >
          <div className={`p-4 rounded-lg shadow-lg backdrop-blur-lg border ${
            notification.type === 'success' ? 'bg-green-500/20 border-green-500/50 text-green-300' :
            notification.type === 'error' ? 'bg-red-500/20 border-red-500/50 text-red-300' :
            'bg-blue-500/20 border-blue-500/50 text-blue-300'
          }`}>
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                {notification.type === 'success' && <CheckCircle size={20} />}
                {notification.type === 'error' && <XCircle size={20} />}
                {notification.type === 'info' && <AlertCircle size={20} />}
                <span className="text-sm font-medium">{notification.message}</span>
              </div>
              <button
                onClick={() => setNotification(null)}
                className="text-white/70 hover:text-white"
              >
                <X size={16} />
              </button>
            </div>
          </div>
        </motion.div>
      )}

      {/* Animated Background Circles */}
      <motion.div
        className="absolute w-[600px] h-[600px] bg-purple-500/30 rounded-full blur-3xl"
        animate={{ x: [0, 200, -200, 0], y: [0, -100, 100, 0] }}
        transition={{ repeat: Infinity, duration: 20 }}
      />
      <motion.div
        className="absolute right-10 top-20 w-[400px] h-[400px] bg-blue-500/20 rounded-full blur-3xl"
        animate={{ scale: [1, 1.2, 1] }}
        transition={{ repeat: Infinity, duration: 8 }}
      />

      {/* Custom Cursor */}
      <div
        ref={cursorRef}
        className="fixed w-8 h-8 rounded-full bg-purple-400/50 border border-purple-300/70 pointer-events-none transform -translate-x-1/2 -translate-y-1/2 blur-sm shadow-xl z-40"
      ></div>

      {/* Loading Overlay */}
      {(loading || exportLoading) && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center">
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 text-center">
            <RefreshCw className="animate-spin h-16 w-16 text-purple-500 mx-auto mb-4" />
            <p className="text-xl">{exportLoading ? 'Exporting Results...' : 'Processing Request...'}</p>
          </div>
        </div>
      )}

      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <motion.div
          className="text-center mb-12"
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1.2 }}
        >
          <h1 className="text-4xl md:text-6xl font-extrabold mb-6 tracking-wide">
            Enhanced OMR Scanner
          </h1>
          
          {/* Status Indicators */}
          <div className="flex flex-wrap justify-center gap-4 mb-8">
            <div className={`px-6 py-3 rounded-full text-sm font-medium ${
              answerKeyLoaded ? 'bg-green-500/20 text-green-300 border border-green-500/30' : 'bg-red-500/20 text-red-300 border border-red-500/30'
            }`}>
              <div className="flex items-center space-x-2">
                {answerKeyLoaded ? <CheckCircle size={16} /> : <XCircle size={16} />}
                <span>Answer Key: {answerKeyLoaded ? `Loaded (${Object.keys(answerKey || {}).length} questions)` : 'Not Loaded'}</span>
              </div>
            </div>
            <div className={`px-6 py-3 rounded-full text-sm font-medium ${
              (results && results.length > 0) ? 'bg-green-500/20 text-green-300 border border-green-500/30' : 'bg-gray-500/20 text-gray-300 border border-gray-500/30'
            }`}>
              <div className="flex items-center space-x-2">
                <FileText size={16} />
                <span>Results: {results ? results.length : 0} Scanned</span>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Student Details Input */}
        <motion.div 
          className="mb-12"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <Card className="max-w-4xl mx-auto">
            <CardContent>
              <h3 className="text-xl font-bold mb-6 text-center">Student Information</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="flex items-center space-x-3">
                  <User size={20} className="text-purple-400 flex-shrink-0" />
                  <input
                    type="text"
                    value={studentName}
                    onChange={(e) => setStudentName(e.target.value)}
                    className="w-full p-4 rounded-lg text-black bg-white/90 backdrop-blur-sm shadow-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                    placeholder="Student Name"
                  />
                </div>
                <div className="flex items-center space-x-3">
                  <IdCard size={20} className="text-purple-400 flex-shrink-0" />
                  <input
                    type="text"
                    value={regNo}
                    onChange={(e) => setRegNo(e.target.value)}
                    className="w-full p-4 rounded-lg text-black bg-white/90 backdrop-blur-sm shadow-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                    placeholder="Registration Number"
                  />
                </div>
                <div className="flex items-center space-x-3">
                  <GraduationCap size={20} className="text-purple-400 flex-shrink-0" />
                  <input
                    type="text"
                    value={studentClass}
                    onChange={(e) => setStudentClass(e.target.value)}
                    className="w-full p-4 rounded-lg text-black bg-white/90 backdrop-blur-sm shadow-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                    placeholder="Class"
                  />
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Settings Panel */}
        {showSettings && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="mb-12"
          >
            <Card className="max-w-4xl mx-auto">
              <CardContent>
                <div className="flex justify-between items-center mb-6">
                  <h3 className="text-xl font-bold">Settings & Configuration</h3>
                  <button
                    onClick={() => setShowSettings(false)}
                    className="text-gray-400 hover:text-white"
                  >
                    <X size={24} />
                  </button>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-medium mb-3">IP Camera URL:</label>
                    <input
                      type="text"
                      value={ipCamUrl}
                      onChange={(e) => setIpCamUrl(e.target.value)}
                      className="w-full p-4 rounded-lg text-black bg-white/90 backdrop-blur-sm shadow-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                      placeholder="http://192.168.1.100:8080/shot.jpg"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-3">Export Format:</label>
                    <select
                      value={selectedFormat}
                      onChange={(e) => setSelectedFormat(e.target.value)}
                      className="w-full p-4 rounded-lg text-black bg-white/90 backdrop-blur-sm shadow-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                    >
                      <option value="pdf">PDF Document</option>
                      <option value="xlsx">Excel Spreadsheet</option>
                      <option value="csv">CSV File</option>
                      <option value="word">Word Document</option>
                      <option value="txt">Text File</option>
                    </select>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}

        {/* Main Action Cards */}
        <motion.div 
          className="mb-12"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
        >
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-4 gap-6">
            
            {/* Upload Single OMR */}
            <motion.div
              whileHover={{ scale: 1.05, y: -5 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleUploadOMR}
              className="cursor-pointer"
            >
              <Card className="h-full hover:bg-white/15 transition-all duration-300">
                <CardContent className="text-center">
                  <div className="flex justify-center mb-4 text-purple-400">
                    <Upload size={32} />
                  </div>
                  <h3 className="text-lg font-bold mb-2">Upload Single</h3>
                  <p className="text-sm text-gray-300">Scan one OMR sheet</p>
                </CardContent>
              </Card>
            </motion.div>

            {/* Upload Multiple OMR */}
            <motion.div
              whileHover={{ scale: 1.05, y: -5 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleUploadMultipleOMR}
              className="cursor-pointer"
            >
              <Card className="h-full hover:bg-white/15 transition-all duration-300">
                <CardContent className="text-center">
                  <div className="flex justify-center mb-4 text-green-400">
                    <Users size={32} />
                  </div>
                  <h3 className="text-lg font-bold mb-2">Multiple Scan</h3>
                  <p className="text-sm text-gray-300">Batch process sheets</p>
                </CardContent>
              </Card>
            </motion.div>

            {/* Camera Scan */}
            <motion.div
              whileHover={{ scale: 1.05, y: -5 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleCameraScan}
              className="cursor-pointer"
            >
              <Card className="h-full hover:bg-white/15 transition-all duration-300">
                <CardContent className="text-center">
                  <div className="flex justify-center mb-4 text-blue-400">
                    <Camera size={32} />
                  </div>
                  <h3 className="text-lg font-bold mb-2">Live Camera</h3>
                  <p className="text-sm text-gray-300">Instant scanning</p>
                </CardContent>
              </Card>
            </motion.div>

            {/* View Results */}
            <motion.div
              whileHover={{ scale: 1.05, y: -5 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setShowResults(!showResults)}
              className="cursor-pointer"
            >
              <Card className="h-full hover:bg-white/15 transition-all duration-300">
                <CardContent className="text-center">
                  <div className="flex justify-center mb-4 text-cyan-400">
                    <Eye size={32} />
                  </div>
                  <h3 className="text-lg font-bold mb-2">View Results</h3>
                  <p className="text-sm text-gray-300">Show all scans</p>
                </CardContent>
              </Card>
            </motion.div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-4 gap-6 mt-6">
            {/* Upload Answer Key */}
            <motion.div
              whileHover={{ scale: 1.05, y: -5 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleUploadAnswerKey}
              className="cursor-pointer"
            >
              <Card className="h-full hover:bg-white/15 transition-all duration-300">
                <CardContent className="text-center">
                  <div className="flex justify-center mb-4 text-orange-400">
                    <FileSpreadsheet size={32} />
                  </div>
                  <h3 className="text-lg font-bold mb-2">Upload Key</h3>
                  <p className="text-sm text-gray-300">JSON/CSV file</p>
                </CardContent>
              </Card>
            </motion.div>

            {/* Scan Answer Key */}
            <motion.div
              whileHover={{ scale: 1.05, y: -5 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleScanAnswerKeyFromCamera}
              className="cursor-pointer"
            >
              <Card className="h-full hover:bg-white/15 transition-all duration-300">
                <CardContent className="text-center">
                  <div className="flex justify-center mb-4 text-yellow-400">
                    <ScanLine size={32} />
                  </div>
                  <h3 className="text-lg font-bold mb-2">Scan Key</h3>
                  <p className="text-sm text-gray-300">From camera</p>
                </CardContent>
              </Card>
            </motion.div>

            {/* Settings */}
            <motion.div
              whileHover={{ scale: 1.05, y: -5 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setShowSettings(!showSettings)}
              className="cursor-pointer"
            >
              <Card className="h-full hover:bg-white/15 transition-all duration-300">
                <CardContent className="text-center">
                  <div className="flex justify-center mb-4 text-gray-400">
                    <Settings size={32} />
                  </div>
                  <h3 className="text-lg font-bold mb-2">Settings</h3>
                  <p className="text-sm text-gray-300">Configure app</p>
                </CardContent>
              </Card>
            </motion.div>

            {/* Clear All */}
            <motion.div
              whileHover={{ scale: 1.05, y: -5 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleClearAllResults}
              className="cursor-pointer"
            >
              <Card className="h-full hover:bg-white/15 transition-all duration-300">
                <CardContent className="text-center">
                  <div className="flex justify-center mb-4 text-red-400">
                    <Trash2 size={32} />
                  </div>
                  <h3 className="text-lg font-bold mb-2">Clear All</h3>
                  <p className="text-sm text-gray-300">Delete all results</p>
                </CardContent>
              </Card>
            </motion.div>
          </div>
        </motion.div>

        {/* Results Display */}
        {showResults && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-12"
          >
            <Card className="max-w-7xl mx-auto">
              <CardContent>
                <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 gap-4">
                  <div>
                    <h3 className="text-2xl font-bold">Scan Results</h3>
                    <p className="text-gray-400">Total: {results.length} scanned sheets</p>
                  </div>
                  <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
                    <div className="flex items-center space-x-2">
                      <label className="text-sm font-medium">Export Format:</label>
                      <select
                        value={selectedFormat}
                        onChange={(e) => setSelectedFormat(e.target.value)}
                        className="p-2 rounded-lg text-black bg-white/90 focus:outline-none focus:ring-2 focus:ring-purple-500"
                      >
                        <option value="pdf">PDF</option>
                        <option value="xlsx">Excel</option>
                        <option value="csv">CSV</option>
                        <option value="word">Word</option>
                        <option value="txt">Text</option>
                      </select>
                    </div>
                    <button
                      onClick={() => setShowResults(false)}
                      className="text-gray-400 hover:text-white"
                    >
                      <X size={24} />
                    </button>
                  </div>
                </div>
                
                {results.length > 0 ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-h-96 overflow-y-auto">
                    {results.map((result, index) => (
                      <motion.div
                        key={result.id || index}
                        whileHover={{ scale: 1.02 }}
                        className="bg-white/5 rounded-lg p-6 border border-white/10 hover:border-white/20 transition-all duration-300"
                      >
                        <div className="flex justify-between items-start mb-4">
                          <div className="flex-1">
                            <h4 className="font-semibold text-lg mb-1">
                              {result.student_name || `Student ${index + 1}`}
                            </h4>
                            <p className="text-sm text-gray-300">
                              Reg: {result.reg_no || 'N/A'}
                            </p>
                            <p className="text-sm text-gray-300">
                              Class: {result.class || 'N/A'}
                            </p>
                            {result.timestamp && (
                              <p className="text-xs text-gray-400 mt-1">
                                {new Date(result.timestamp).toLocaleDateString()}
                              </p>
                            )}
                          </div>
                          <div className="text-right">
                            <div className="text-2xl font-bold text-green-400 mb-1">
                              {result.score || 0}/{result.total_questions || result.total || 0}
                            </div>
                            <div className="text-sm text-gray-300">
                              {(result.percentage || 0).toFixed(1)}%
                            </div>
                          </div>
                        </div>
                        
                        <div className="grid grid-cols-2 gap-3 text-xs mb-4">
                          <div className="flex items-center space-x-1">
                            <CheckCircle size={12} className="text-green-400" />
                            <span className="text-green-400">Correct: {result.correct_answers || 0}</span>
                          </div>
                          <div className="flex items-center space-x-1">
                            <XCircle size={12} className="text-red-400" />
                            <span className="text-red-400">Wrong: {result.wrong_answers || 0}</span>
                          </div>
                          <div className="flex items-center space-x-1">
                            <AlertCircle size={12} className="text-yellow-400" />
                            <span className="text-yellow-400">Blank: {result.blank_answers || 0}</span>
                          </div>
                          <div className="flex items-center space-x-1">
                            <AlertCircle size={12} className="text-gray-400" />
                            <span className="text-gray-400">Multiple: {result.multiple_answers || 0}</span>
                          </div>
                        </div>

                        {/* Answer Pattern Display */}
                        {result.detected_answers && (
                          <div className="mb-4">
                            <p className="text-xs text-gray-400 mb-2">Detected Answers:</p>
                            <div className="text-xs font-mono bg-black/30 p-3 rounded max-h-16 overflow-y-auto">
                              {typeof result.detected_answers === 'object' 
                                ? Object.entries(result.detected_answers).map(([q, ans]) => `Q${q}:${ans}`).join(' ')
                                : result.detected_answers}
                            </div>
                          </div>
                        )}

                        <div className="flex flex-col sm:flex-row gap-2">
                          <button
                            onClick={() => handleExportResult(index)}
                            disabled={exportLoading}
                            className="flex-1 bg-blue-500/20 hover:bg-blue-500/30 text-blue-300 px-4 py-2 rounded-lg text-sm transition-colors disabled:opacity-50 flex items-center justify-center space-x-1"
                          >
                            <Download size={14} />
                            <span>Export {selectedFormat.toUpperCase()}</span>
                          </button>
                          <button
                            onClick={() => setCurrentResult(result)}
                            className="bg-green-500/20 hover:bg-green-500/30 text-green-300 px-4 py-2 rounded-lg text-sm transition-colors flex items-center justify-center space-x-1"
                          >
                            <Eye size={14} />
                            <span>View</span>
                          </button>
                          <button
                            onClick={() => handleDeleteResult(index)}
                            className="bg-red-500/20 hover:bg-red-500/30 text-red-300 px-4 py-2 rounded-lg text-sm transition-colors flex items-center justify-center"
                          >
                            <Trash2 size={14} />
                          </button>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <FileText size={48} className="mx-auto mb-4 text-gray-400" />
                    <h4 className="text-xl font-semibold mb-2">No Results Yet</h4>
                    <p className="text-gray-400 max-w-md mx-auto">
                      Scan some OMR sheets to see results here. Make sure to load an answer key first!
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>
        )}

        {/* Current Result Detail View */}
        {currentResult && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="mb-12"
          >
            <Card className="max-w-6xl mx-auto">
              <CardContent>
                <div className="flex justify-between items-center mb-6">
                  <h3 className="text-2xl font-bold">Detailed Result View</h3>
                  <button
                    onClick={() => setCurrentResult(null)}
                    className="text-gray-400 hover:text-white transition-colors"
                  >
                    <X size={24} />
                  </button>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  {/* Student Information */}
                  <div className="space-y-6">
                    <div>
                      <h4 className="text-lg font-semibold mb-4 text-purple-400">Student Information</h4>
                      <div className="bg-white/5 rounded-lg p-6 space-y-3">
                        <div className="flex items-center space-x-2">
                          <User size={16} className="text-gray-400" />
                          <span className="text-gray-300">Name:</span>
                          <span className="font-medium">{currentResult.student_name || 'N/A'}</span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <IdCard size={16} className="text-gray-400" />
                          <span className="text-gray-300">Registration:</span>
                          <span className="font-medium">{currentResult.reg_no || 'N/A'}</span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <GraduationCap size={16} className="text-gray-400" />
                          <span className="text-gray-300">Class:</span>
                          <span className="font-medium">{currentResult.class || 'N/A'}</span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <AlertCircle size={16} className="text-gray-400" />
                          <span className="text-gray-300">Scan Time:</span>
                          <span className="font-medium">
                            {currentResult.timestamp ? new Date(currentResult.timestamp).toLocaleString() : new Date().toLocaleString()}
                          </span>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h4 className="text-lg font-semibold mb-4 text-green-400">Score Summary</h4>
                      <div className="bg-white/5 rounded-lg p-6">
                        <div className="text-center mb-6">
                          <div className="text-5xl font-bold text-green-400 mb-2">
                            {currentResult.score || 0}/{currentResult.total_questions || currentResult.total || 0}
                          </div>
                          <div className="text-2xl text-gray-300 mb-4">
                            {(currentResult.percentage || 0).toFixed(1)}%
                          </div>
                          <div className={`inline-block px-4 py-2 rounded-full text-sm font-medium ${
                            (currentResult.percentage || 0) >= 80 ? 'bg-green-500/20 text-green-300' :
                            (currentResult.percentage || 0) >= 60 ? 'bg-yellow-500/20 text-yellow-300' :
                            'bg-red-500/20 text-red-300'
                          }`}>
                            {(currentResult.percentage || 0) >= 80 ? 'Excellent' :
                             (currentResult.percentage || 0) >= 60 ? 'Good' : 'Needs Improvement'}
                          </div>
                        </div>
                        
                        <div className="grid grid-cols-2 gap-4">
                          <div className="text-center bg-green-500/10 rounded-lg p-4">
                            <div className="text-2xl font-bold text-green-400">{currentResult.correct_answers || 0}</div>
                            <div className="text-sm text-green-300">Correct</div>
                          </div>
                          <div className="text-center bg-red-500/10 rounded-lg p-4">
                            <div className="text-2xl font-bold text-red-400">{currentResult.wrong_answers || 0}</div>
                            <div className="text-sm text-red-300">Wrong</div>
                          </div>
                          <div className="text-center bg-yellow-500/10 rounded-lg p-4">
                            <div className="text-2xl font-bold text-yellow-400">{currentResult.blank_answers || 0}</div>
                            <div className="text-sm text-yellow-300">Blank</div>
                          </div>
                          <div className="text-center bg-gray-500/10 rounded-lg p-4">
                            <div className="text-2xl font-bold text-gray-400">{currentResult.multiple_answers || 0}</div>
                            <div className="text-sm text-gray-300">Multiple</div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Detailed Answers */}
                  <div>
                    <h4 className="text-lg font-semibold mb-4 text-blue-400">Answer Analysis</h4>
                    <div className="bg-white/5 rounded-lg p-6 max-h-96 overflow-y-auto">
                      {currentResult.question_details ? (
                        <div className="space-y-2">
                          {currentResult.question_details.map((detail, index) => (
                            <div 
                              key={index} 
                              className={`flex justify-between items-center p-3 rounded-lg text-sm transition-all ${
                                detail.is_correct || detail.status === '✓' ? 'bg-green-500/10 text-green-300 border border-green-500/20' : 
                                detail.detected === 'N/A' || detail.detected === 'BLANK' || detail.status === '-' ? 'bg-yellow-500/10 text-yellow-300 border border-yellow-500/20' :
                                detail.detected === 'MULTIPLE' ? 'bg-gray-500/10 text-gray-300 border border-gray-500/20' :
                                'bg-red-500/10 text-red-300 border border-red-500/20'
                              }`}
                            >
                              <span className="font-medium">Q{detail.question || index + 1}:</span>
                              <div className="flex items-center space-x-3">
                                <span>
                                  <span className="text-gray-400">Detected:</span> {detail.detected || detail.detected_answer || 'N/A'}
                                </span>
                                <span>
                                  <span className="text-gray-400">Correct:</span> {detail.correct || detail.correct_answer || 'N/A'}
                                </span>
                                <span className="text-lg">
                                  {detail.is_correct || detail.status === '✓' ? '✓' : 
                                   detail.detected === 'N/A' || detail.detected === 'BLANK' || detail.status === '-' ? '○' :
                                   detail.detected === 'MULTIPLE' ? '?' : '✗'}
                                </span>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : currentResult.detected_answers ? (
                        <div className="space-y-4">
                          <div>
                            <p className="text-gray-300 mb-3 font-medium">Detected Answers:</p>
                            <div className="bg-black/30 p-4 rounded-lg font-mono text-sm">
                              {typeof currentResult.detected_answers === 'object' 
                                ? Object.entries(currentResult.detected_answers).map(([q, ans], i) => (
                                    <span key={i} className="inline-block mr-4 mb-2">
                                      Q{q}: <span className="text-green-400">{ans}</span>
                                    </span>
                                  ))
                                : currentResult.detected_answers}
                            </div>
                          </div>
                          {answerKey && (
                            <div>
                              <p className="text-gray-300 mb-3 font-medium">Correct Answers:</p>
                              <div className="bg-black/30 p-4 rounded-lg font-mono text-sm">
                                {Object.entries(answerKey).map(([q, ans], i) => (
                                  <span key={i} className="inline-block mr-4 mb-2">
                                    Q{q}: <span className="text-blue-400">{ans}</span>
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      ) : (
                        <div className="text-center py-8">
                          <AlertCircle size={48} className="mx-auto mb-4 text-gray-400" />
                          <p className="text-gray-400">No detailed answer data available</p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="mt-8 flex flex-col sm:flex-row justify-center gap-4">
                  <button
                    onClick={() => {
                      const resultIndex = results.findIndex(r => r.id === currentResult.id || r === currentResult);
                      if (resultIndex >= 0) handleExportResult(resultIndex);
                    }}
                    disabled={exportLoading}
                    className="bg-blue-500/20 hover:bg-blue-500/30 text-blue-300 px-8 py-3 rounded-lg transition-colors disabled:opacity-50 flex items-center justify-center space-x-2"
                  >
                    <Download size={20} />
                    <span>Export as {selectedFormat.toUpperCase()}</span>
                  </button>
                  <button
                    onClick={() => setCurrentResult(null)}
                    className="bg-gray-500/20 hover:bg-gray-500/30 text-gray-300 px-8 py-3 rounded-lg transition-colors flex items-center justify-center space-x-2"
                  >
                    <X size={20} />
                    <span>Close</span>
                  </button>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}

        {/* Footer */}
        <div className="text-center text-gray-400 text-sm mt-12">
          <p className="mb-2">Enhanced OMR Scanner - Scan, Process, Export</p>
          <p className="flex items-center justify-center space-x-2">
            {answerKeyLoaded ? (
              <><CheckCircle size={16} className="text-green-400" /><span>Ready to scan</span></>
            ) : (
              <><XCircle size={16} className="text-red-400" /><span>Load answer key to begin</span></>
            )}
          </p>
        </div>
      </div>
    </div>
  );
}