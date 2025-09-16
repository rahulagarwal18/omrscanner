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
  CreditCard,
  GraduationCap,
  Eye,
  Trash2,
  Settings,
  CheckCircle,
  XCircle,
  AlertCircle,
  RefreshCw,
  X,
  AlertTriangle,
  Database,
  Filter
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
  const [serverAwake, setServerAwake] = useState(false);
  const [serverChecking, setServerChecking] = useState(true);
  
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
  const [showDatabase, setShowDatabase] = useState(false);
  const [notification, setNotification] = useState(null);
  
  // Database view states
  const [databaseStats, setDatabaseStats] = useState(null);
  const [filterClass, setFilterClass] = useState("");
  const [filterDate, setFilterDate] = useState("");
  const [sortBy, setSortBy] = useState("timestamp");
  const [sortOrder, setSortOrder] = useState("desc");

  // UPDATED FOR RENDER: Dynamic API URL based on environment
  const API_BASE_URL = window.location.hostname === 'localhost' 
    ? "http://localhost:5000/api"
    : "/api";  // For production on Render

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

  // ADDED FOR RENDER: Check if backend server is awake
  useEffect(() => {
    const checkBackend = async () => {
      setServerChecking(true);
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout
        
        const response = await fetch(`${API_BASE_URL}/health`, {
          signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        if (response.ok) {
          setServerAwake(true);
          setServerChecking(false);
          // Only fetch data after server is confirmed awake
          fetchAllResults();
          checkAnswerKey();
        } else {
          throw new Error('Server not ready');
        }
      } catch (err) {
        if (err.name === 'AbortError') {
          showNotification("Server is taking longer than usual to start. Please wait...", "warning");
        } else if (window.location.hostname !== 'localhost') {
          // Only show wake-up message in production
          showNotification("🌙 Server is waking up... This usually takes 30-50 seconds on free hosting", "info");
        }
        
        // Retry after 5 seconds
        setTimeout(() => {
          checkBackend();
        }, 5000);
      }
    };
    
    checkBackend();
  }, []);

  const checkAnswerKey = async () => {
    if (!serverAwake) return;
    
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
    if (!serverAwake) return;
    
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
      if (serverAwake) {
        showNotification("Cannot fetch results. Please check your connection.", "error");
      }
    }
  };

  // Fetch database statistics
  const fetchDatabaseStats = async () => {
    if (!serverAwake) {
      showNotification("Server is still waking up. Please wait...", "info");
      return;
    }
    
    try {
      const response = await fetch(`${API_BASE_URL}/database/stats`);
      if (response.ok) {
        const data = await response.json();
        setDatabaseStats(data);
      }
    } catch (err) {
      console.error("Failed to fetch database stats:", err);
    }
  };

  // Export all results to CSV
  const handleExportAllToCSV = async () => {
    if (!serverAwake) {
      showNotification("Server is still waking up. Please wait...", "info");
      return;
    }
    
    setExportLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/database/export/csv`);
      if (!response.ok) throw new Error('Export failed');
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `all_results_${new Date().toISOString().split('T')[0]}.csv`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
      showNotification("All results exported to CSV successfully!", "success");
    } catch (err) {
      showNotification("Failed to export database", "error");
    } finally {
      setExportLoading(false);
    }
  };

  // Export all results to Excel
  const handleExportAllToExcel = async () => {
    if (!serverAwake) {
      showNotification("Server is still waking up. Please wait...", "info");
      return;
    }
    
    setExportLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/database/export/excel`);
      if (!response.ok) throw new Error('Export failed');
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `all_results_${new Date().toISOString().split('T')[0]}.xlsx`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
      showNotification("All results exported to Excel successfully!", "success");
    } catch (err) {
      showNotification("Failed to export database", "error");
    } finally {
      setExportLoading(false);
    }
  };

  // Get filtered results
  const getFilteredResults = () => {
    let filtered = [...results];
    
    // Filter by class
    if (filterClass) {
      filtered = filtered.filter(r => 
        r.class && r.class.toLowerCase().includes(filterClass.toLowerCase())
      );
    }
    
    // Filter by date
    if (filterDate) {
      filtered = filtered.filter(r => {
        const resultDate = new Date(r.timestamp).toDateString();
        const filterDateObj = new Date(filterDate).toDateString();
        return resultDate === filterDateObj;
      });
    }
    
    // Sort results
    filtered.sort((a, b) => {
      let aVal = a[sortBy];
      let bVal = b[sortBy];
      
      if (sortBy === 'timestamp') {
        aVal = new Date(aVal);
        bVal = new Date(bVal);
      } else if (sortBy === 'percentage' || sortBy === 'score') {
        aVal = parseFloat(aVal) || 0;
        bVal = parseFloat(bVal) || 0;
      }
      
      if (sortOrder === 'asc') {
        return aVal > bVal ? 1 : -1;
      } else {
        return aVal < bVal ? 1 : -1;
      }
    });
    
    return filtered;
  };

  // Calculate statistics
  const calculateStats = (resultsList) => {
    if (!resultsList || resultsList.length === 0) return null;
    
    const avgScore = resultsList.reduce((sum, r) => sum + (r.percentage || 0), 0) / resultsList.length;
    const highestScore = Math.max(...resultsList.map(r => r.percentage || 0));
    const lowestScore = Math.min(...resultsList.map(r => r.percentage || 0));
    const totalMultiple = resultsList.reduce((sum, r) => sum + (r.multiple_answers || 0), 0);
    
    const classStats = {};
    resultsList.forEach(r => {
      const cls = r.class || 'N/A';
      if (!classStats[cls]) {
        classStats[cls] = { count: 0, totalScore: 0 };
      }
      classStats[cls].count++;
      classStats[cls].totalScore += r.percentage || 0;
    });
    
    return {
      avgScore,
      highestScore,
      lowestScore,
      totalMultiple,
      classStats
    };
  };

  // Clear all results
  const handleClearAllResults = async () => {
    if (!serverAwake) {
      showNotification("Server is still waking up. Please wait...", "info");
      return;
    }
    
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

  // Scan Answer Key from Camera
  const handleScanAnswerKeyFromCamera = async () => {
    if (!serverAwake) {
      showNotification("Server is still waking up. Please wait...", "info");
      return;
    }
    
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

  // Set Sample Answer Key
  const handleSetSampleAnswerKey = async () => {
    if (!serverAwake) {
      showNotification("Server is still waking up. Please wait...", "info");
      return;
    }
    
    setLoading(true);
    
    try {
      const response = await fetch(`${API_BASE_URL}/set-sample-answer-key`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        setAnswerKey(data.answer_key);
        setAnswerKeyLoaded(true);
        showNotification("Sample answer key loaded successfully!", "success");
        console.log("Sample answer key set:", data.answer_key);
      } else {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to set sample answer key');
      }
      
    } catch (err) {
      console.error("Sample answer key error:", err);
      showNotification(err.message, "error");
    } finally {
      setLoading(false);
    }
  };

  // Upload Answer Key from File
  const handleUploadAnswerKey = () => {
    if (!serverAwake) {
      showNotification("Server is still waking up. Please wait...", "info");
      return;
    }
    answerKeyRef.current?.click();
  };
  
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
  const handleUploadOMR = () => {
    if (!serverAwake) {
      showNotification("Server is still waking up. Please wait...", "info");
      return;
    }
    fileInputRef.current?.click();
  };
  
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
        
        // Check for multiple answers and show special notification
        if (data.multiple_answers && data.multiple_answers > 0) {
          showNotification(`⚠️ Detected ${data.multiple_answers} questions with multiple bubbles marked!`, "warning");
        } else {
          showNotification("OMR sheet scanned successfully!", "success");
        }
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
  const handleUploadMultipleOMR = () => {
    if (!serverAwake) {
      showNotification("Server is still waking up. Please wait...", "info");
      return;
    }
    multipleFileInputRef.current?.click();
  };
  
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
        
        // Count total multiple answers across all sheets
        const totalMultiple = data.results?.reduce((sum, r) => sum + (r.multiple_answers || 0), 0) || 0;
        if (totalMultiple > 0) {
          showNotification(`⚠️ Processed ${data.total_processed} sheets. Found ${totalMultiple} questions with multiple marks!`, "warning");
        } else {
          showNotification(`Successfully processed ${data.total_processed} OMR sheets!`, "success");
        }
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
    if (!serverAwake) {
      showNotification("Server is still waking up. Please wait...", "info");
      return;
    }
    
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
        
        // Check for multiple answers and show special notification
        if (data.multiple_answers && data.multiple_answers > 0) {
          showNotification(`⚠️ Detected ${data.multiple_answers} questions with multiple bubbles marked!`, "warning");
        } else {
          showNotification("Camera scan completed successfully!", "success");
        }
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

  // Export Results function
  const handleExportResult = async (resultId) => {
    if (!serverAwake) {
      showNotification("Server is still waking up. Please wait...", "info");
      return;
    }
    
    // Handle both direct ID and index-based calls
    let actualResultId;
    
    if (typeof resultId === 'object') {
      // If passed a result object, extract the ID
      actualResultId = resultId.id;
    } else if (typeof resultId === 'number') {
      // It's already an ID
      actualResultId = resultId;
    } else {
      actualResultId = resultId;
    }

    if (!actualResultId) {
      showNotification("Invalid result selection", "error");
      return;
    }
    
    setExportLoading(true);
    
    try {
      console.log("Exporting result ID:", actualResultId, "Format:", selectedFormat);
      
      // Call the export endpoint with the actual result ID
      const response = await fetch(`${API_BASE_URL}/export/${selectedFormat}/${actualResultId}`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `Export failed: ${response.statusText}`);
      }

      const data = await response.json();
      console.log("Export response:", data);

      if (data.success && data.filename) {
        // Create the download URL
        const downloadUrl = `${API_BASE_URL}/download/${data.filename}`;
        
        // Create a temporary link and trigger download
        const link = document.createElement('a');
        link.href = downloadUrl;
        link.download = data.filename;
        link.style.display = 'none';
        
        document.body.appendChild(link);
        link.click();
        
        // Clean up
        setTimeout(() => {
          document.body.removeChild(link);
        }, 100);
        
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
  const handleDeleteResult = async (resultId) => {
    if (!serverAwake) {
      showNotification("Server is still waking up. Please wait...", "info");
      return;
    }
    
    if (!confirm("Are you sure you want to delete this result?")) return;
    
    try {
      const response = await fetch(`${API_BASE_URL}/results/${resultId}`, {
        method: 'DELETE'
      });
      
      if (response.ok) {
        await fetchAllResults();
        if (currentResult && currentResult.id === resultId) {
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

      {/* ADDED: Server Status Banner for Production */}
      {window.location.hostname !== 'localhost' && !serverAwake && (
        <div className="bg-yellow-500/20 border-b border-yellow-500/50 p-3 text-center">
          <div className="flex items-center justify-center space-x-2">
            {serverChecking ? (
              <>
                <RefreshCw className="animate-spin h-4 w-4" />
                <span className="text-sm">Server is starting up... This is normal for free hosting (30-50 seconds)</span>
              </>
            ) : (
              <span className="text-sm">Connecting to server...</span>
            )}
          </div>
        </div>
      )}

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
            notification.type === 'warning' ? 'bg-yellow-500/20 border-yellow-500/50 text-yellow-300' :
            'bg-blue-500/20 border-blue-500/50 text-blue-300'
          }`}>
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                {notification.type === 'success' && <CheckCircle size={20} />}
                {notification.type === 'error' && <XCircle size={20} />}
                {notification.type === 'warning' && <AlertTriangle size={20} />}
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
      {(loading || exportLoading || (serverChecking && !serverAwake)) && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center">
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 text-center">
            <RefreshCw className="animate-spin h-16 w-16 text-purple-500 mx-auto mb-4" />
            <p className="text-xl">
              {serverChecking && !serverAwake ? 'Connecting to Server...' : 
               exportLoading ? 'Exporting Results...' : 
               'Processing Request...'}
            </p>
            {serverChecking && !serverAwake && window.location.hostname !== 'localhost' && (
              <p className="text-sm text-gray-400 mt-2">Free hosting may take 30-50 seconds to wake up</p>
            )}
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
          <div className="text-sm text-gray-400 mb-4">
            v7.0 - Multiple Bubble Detection Fixed
            {window.location.hostname !== 'localhost' && (
              <span className="ml-2 text-yellow-400">(Running on Free Hosting)</span>
            )}
          </div>
          
          {/* Status Indicators */}
          <div className="flex flex-wrap justify-center gap-4 mb-8">
            <div className={`px-6 py-3 rounded-full text-sm font-medium ${
              serverAwake ? 'bg-green-500/20 text-green-300 border border-green-500/30' : 'bg-yellow-500/20 text-yellow-300 border border-yellow-500/30'
            }`}>
              <div className="flex items-center space-x-2">
                {serverAwake ? <CheckCircle size={16} /> : <AlertCircle size={16} />}
                <span>Server: {serverAwake ? 'Connected' : 'Connecting...'}</span>
              </div>
            </div>
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

          {/* Quick Action: Load Sample Answer Key if not loaded */}
          {!answerKeyLoaded && serverAwake && (
            <div className="mb-4">
              <button
                onClick={handleSetSampleAnswerKey}
                className="bg-orange-500/20 hover:bg-orange-500/30 text-orange-300 px-6 py-3 rounded-lg transition-colors border border-orange-500/30"
              >
                Load Sample Answer Key (10 Questions)
              </button>
            </div>
          )}
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
                    disabled={!serverAwake}
                  />
                </div>
                <div className="flex items-center space-x-3">
                  <CreditCard size={20} className="text-purple-400 flex-shrink-0" />
                  <input
                    type="text"
                    value={regNo}
                    onChange={(e) => setRegNo(e.target.value)}
                    className="w-full p-4 rounded-lg text-black bg-white/90 backdrop-blur-sm shadow-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                    placeholder="Registration Number"
                    disabled={!serverAwake}
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
                    disabled={!serverAwake}
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
                      disabled={!serverAwake}
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-3">Export Format:</label>
                    <select
                      value={selectedFormat}
                      onChange={(e) => setSelectedFormat(e.target.value)}
                      className="w-full p-4 rounded-lg text-black bg-white/90 backdrop-blur-sm shadow-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                      disabled={!serverAwake}
                    >
                      <option value="pdf">PDF Document</option>
                      <option value="xlsx">Excel Spreadsheet</option>
                      <option value="csv">CSV File</option>
                      <option value="word">Word Document</option>
                      <option value="txt">Text File</option>
                    </select>
                  </div>
                </div>
                {answerKey && (
                  <div className="mt-6">
                    <h4 className="text-sm font-medium mb-3">Current Answer Key:</h4>
                    <div className="bg-black/30 p-4 rounded-lg">
                      <div className="grid grid-cols-5 gap-2">
                        {Object.entries(answerKey).map(([q, ans]) => (
                          <div key={q} className="bg-white/10 p-2 rounded text-center text-sm">
                            Q{q}: <span className="font-bold text-green-400">{ans}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
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
          <div className={`grid grid-cols-2 md:grid-cols-4 lg:grid-cols-4 gap-6 ${!serverAwake ? 'opacity-50 pointer-events-none' : ''}`}>
            
            {/* Upload Single OMR */}
            <motion.div
              whileHover={{ scale: serverAwake ? 1.05 : 1, y: serverAwake ? -5 : 0 }}
              whileTap={{ scale: serverAwake ? 0.95 : 1 }}
              onClick={handleUploadOMR}
              className={`cursor-pointer ${!serverAwake ? 'cursor-not-allowed' : ''}`}
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
              whileHover={{ scale: serverAwake ? 1.05 : 1, y: serverAwake ? -5 : 0 }}
              whileTap={{ scale: serverAwake ? 0.95 : 1 }}
              onClick={handleUploadMultipleOMR}
              className={`cursor-pointer ${!serverAwake ? 'cursor-not-allowed' : ''}`}
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
              whileHover={{ scale: serverAwake ? 1.05 : 1, y: serverAwake ? -5 : 0 }}
              whileTap={{ scale: serverAwake ? 0.95 : 1 }}
              onClick={handleCameraScan}
              className={`cursor-pointer ${!serverAwake ? 'cursor-not-allowed' : ''}`}
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
              whileHover={{ scale: serverAwake ? 1.05 : 1, y: serverAwake ? -5 : 0 }}
              whileTap={{ scale: serverAwake ? 0.95 : 1 }}
              onClick={() => serverAwake && setShowResults(!showResults)}
              className={`cursor-pointer ${!serverAwake ? 'cursor-not-allowed' : ''}`}
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

          <div className={`grid grid-cols-2 md:grid-cols-4 lg:grid-cols-4 gap-6 mt-6 ${!serverAwake ? 'opacity-50 pointer-events-none' : ''}`}>
            {/* Upload Answer Key */}
            <motion.div
              whileHover={{ scale: serverAwake ? 1.05 : 1, y: serverAwake ? -5 : 0 }}
              whileTap={{ scale: serverAwake ? 0.95 : 1 }}
              onClick={handleUploadAnswerKey}
              className={`cursor-pointer ${!serverAwake ? 'cursor-not-allowed' : ''}`}
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
              whileHover={{ scale: serverAwake ? 1.05 : 1, y: serverAwake ? -5 : 0 }}
              whileTap={{ scale: serverAwake ? 0.95 : 1 }}
              onClick={handleScanAnswerKeyFromCamera}
              className={`cursor-pointer ${!serverAwake ? 'cursor-not-allowed' : ''}`}
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
              whileHover={{ scale: serverAwake ? 1.05 : 1, y: serverAwake ? -5 : 0 }}
              whileTap={{ scale: serverAwake ? 0.95 : 1 }}
              onClick={handleClearAllResults}
              className={`cursor-pointer ${!serverAwake ? 'cursor-not-allowed' : ''}`}
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

          {/* New row for Database Viewer */}
          <div className={`grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 gap-6 mt-6 max-w-2xl mx-auto ${!serverAwake ? 'opacity-50 pointer-events-none' : ''}`}>
            {/* Database Viewer */}
            <motion.div
              whileHover={{ scale: serverAwake ? 1.05 : 1, y: serverAwake ? -5 : 0 }}
              whileTap={{ scale: serverAwake ? 0.95 : 1 }}
              onClick={() => {
                if (serverAwake) {
                  setShowDatabase(!showDatabase);
                  if (!showDatabase) fetchDatabaseStats();
                }
              }}
              className={`cursor-pointer ${!serverAwake ? 'cursor-not-allowed' : ''}`}
            >
              <Card className="h-full hover:bg-white/15 transition-all duration-300 bg-indigo-500/10">
                <CardContent className="text-center">
                  <div className="flex justify-center mb-4 text-indigo-400">
                    <Database size={32} />
                  </div>
                  <h3 className="text-lg font-bold mb-2">Database Viewer</h3>
                  <p className="text-sm text-gray-300">View all historical data</p>
                </CardContent>
              </Card>
            </motion.div>

            {/* Export All Data */}
            <motion.div
              whileHover={{ scale: serverAwake ? 1.05 : 1, y: serverAwake ? -5 : 0 }}
              whileTap={{ scale: serverAwake ? 0.95 : 1 }}
              className={`cursor-pointer ${!serverAwake ? 'cursor-not-allowed' : ''}`}
            >
              <Card className="h-full hover:bg-white/15 transition-all duration-300 bg-emerald-500/10">
                <CardContent className="text-center">
                  <div className="flex justify-center mb-4 text-emerald-400">
                    <Download size={32} />
                  </div>
                  <h3 className="text-lg font-bold mb-2">Export Database</h3>
                  <div className="flex gap-2 justify-center mt-2">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        if (serverAwake) handleExportAllToCSV();
                      }}
                      className="bg-blue-500/20 hover:bg-blue-500/30 text-blue-300 px-3 py-1 rounded text-xs"
                      disabled={!serverAwake}
                    >
                      CSV
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        if (serverAwake) handleExportAllToExcel();
                      }}
                      className="bg-green-500/20 hover:bg-green-500/30 text-green-300 px-3 py-1 rounded text-xs"
                      disabled={!serverAwake}
                    >
                      Excel
                    </button>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </div>
        </motion.div>

        {/* Database Viewer Section */}
        {showDatabase && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-12"
          >
            <Card className="max-w-7xl mx-auto">
              <CardContent>
                <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 gap-4">
                  <div>
                    <h3 className="text-2xl font-bold">Database Viewer</h3>
                    <p className="text-gray-400">Complete historical data of all scanned sheets</p>
                  </div>
                  <button
                    onClick={() => setShowDatabase(false)}
                    className="text-gray-400 hover:text-white"
                  >
                    <X size={24} />
                  </button>
                </div>

                {/* Statistics Cards */}
                {(() => {
                  const filteredResults = getFilteredResults();
                  const stats = calculateStats(filteredResults);
                  
                  return (
                    <>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                        <div className="bg-blue-500/20 rounded-lg p-4 text-center">
                          <div className="text-3xl font-bold text-blue-400">{filteredResults.length}</div>
                          <div className="text-sm text-blue-300">Total Sheets</div>
                        </div>
                        <div className="bg-green-500/20 rounded-lg p-4 text-center">
                          <div className="text-3xl font-bold text-green-400">
                            {stats ? stats.avgScore.toFixed(1) : 0}%
                          </div>
                          <div className="text-sm text-green-300">Average Score</div>
                        </div>
                        <div className="bg-purple-500/20 rounded-lg p-4 text-center">
                          <div className="text-3xl font-bold text-purple-400">
                            {stats ? stats.highestScore.toFixed(1) : 0}%
                          </div>
                          <div className="text-sm text-purple-300">Highest Score</div>
                        </div>
                        <div className="bg-orange-500/20 rounded-lg p-4 text-center">
                          <div className="text-3xl font-bold text-orange-400">
                            {stats ? stats.totalMultiple : 0}
                          </div>
                          <div className="text-sm text-orange-300">Multiple Marks</div>
                        </div>
                      </div>

                      {/* Filters and Controls */}
                      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                        <input
                          type="text"
                          value={filterClass}
                          onChange={(e) => setFilterClass(e.target.value)}
                          className="p-3 rounded-lg text-black bg-white/90 focus:outline-none focus:ring-2 focus:ring-purple-500"
                          placeholder="Filter by class..."
                        />
                        <input
                          type="date"
                          value={filterDate}
                          onChange={(e) => setFilterDate(e.target.value)}
                          className="p-3 rounded-lg text-black bg-white/90 focus:outline-none focus:ring-2 focus:ring-purple-500"
                        />
                        <select
                          value={sortBy}
                          onChange={(e) => setSortBy(e.target.value)}
                          className="p-3 rounded-lg text-black bg-white/90 focus:outline-none focus:ring-2 focus:ring-purple-500"
                        >
                          <option value="timestamp">Sort by Date</option>
                          <option value="student_name">Sort by Name</option>
                          <option value="percentage">Sort by Score</option>
                          <option value="class">Sort by Class</option>
                        </select>
                        <select
                          value={sortOrder}
                          onChange={(e) => setSortOrder(e.target.value)}
                          className="p-3 rounded-lg text-black bg-white/90 focus:outline-none focus:ring-2 focus:ring-purple-500"
                        >
                          <option value="desc">Descending</option>
                          <option value="asc">Ascending</option>
                        </select>
                      </div>

                      {/* Class-wise Statistics */}
                      {stats && stats.classStats && Object.keys(stats.classStats).length > 0 && (
                        <div className="mb-6">
                          <h4 className="text-lg font-semibold mb-3">Class-wise Performance</h4>
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                            {Object.entries(stats.classStats).map(([cls, data]) => (
                              <div key={cls} className="bg-white/5 rounded-lg p-3">
                                <div className="font-medium">{cls}</div>
                                <div className="text-sm text-gray-400">
                                  {data.count} students • Avg: {(data.totalScore / data.count).toFixed(1)}%
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Results Table */}
                      <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                          <thead className="bg-white/10">
                            <tr>
                              <th className="p-3 text-left">ID</th>
                              <th className="p-3 text-left">Name</th>
                              <th className="p-3 text-left">Reg No</th>
                              <th className="p-3 text-left">Class</th>
                              <th className="p-3 text-center">Score</th>
                              <th className="p-3 text-center">Percentage</th>
                              <th className="p-3 text-center">Multiple</th>
                              <th className="p-3 text-left">Date</th>
                              <th className="p-3 text-center">Actions</th>
                            </tr>
                          </thead>
                          <tbody>
                            {filteredResults.slice(0, 50).map((result) => (
                              <tr key={result.id} className="border-b border-white/10 hover:bg-white/5">
                                <td className="p-3">{result.id}</td>
                                <td className="p-3 font-medium">{result.student_name || 'N/A'}</td>
                                <td className="p-3">{result.reg_no || 'N/A'}</td>
                                <td className="p-3">{result.class || 'N/A'}</td>
                                <td className="p-3 text-center">
                                  {result.score}/{result.total_questions || result.total}
                                </td>
                                <td className="p-3 text-center">
                                  <span className={`px-2 py-1 rounded text-xs ${
                                    result.percentage >= 80 ? 'bg-green-500/20 text-green-300' :
                                    result.percentage >= 60 ? 'bg-yellow-500/20 text-yellow-300' :
                                    'bg-red-500/20 text-red-300'
                                  }`}>
                                    {(result.percentage || 0).toFixed(1)}%
                                  </span>
                                </td>
                                <td className="p-3 text-center">
                                  {result.multiple_answers > 0 && (
                                    <span className="bg-orange-500/20 text-orange-300 px-2 py-1 rounded text-xs">
                                      {result.multiple_answers}
                                    </span>
                                  )}
                                </td>
                                <td className="p-3 text-xs">
                                  {new Date(result.timestamp).toLocaleDateString()}
                                </td>
                                <td className="p-3">
                                  <div className="flex gap-2 justify-center">
                                    <button
                                      onClick={() => {
                                        setCurrentResult(result);
                                        setShowDatabase(false);
                                      }}
                                      className="text-blue-400 hover:text-blue-300"
                                    >
                                      <Eye size={16} />
                                    </button>
                                    <button
                                      onClick={() => handleExportResult(result.id)}
                                      className="text-green-400 hover:text-green-300"
                                    >
                                      <Download size={16} />
                                    </button>
                                    <button
                                      onClick={() => handleDeleteResult(result.id)}
                                      className="text-red-400 hover:text-red-300"
                                    >
                                      <Trash2 size={16} />
                                    </button>
                                  </div>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                        {filteredResults.length > 50 && (
                          <div className="text-center py-4 text-gray-400">
                            Showing first 50 of {filteredResults.length} results
                          </div>
                        )}
                        {filteredResults.length === 0 && (
                          <div className="text-center py-8 text-gray-400">
                            No results found. Try adjusting your filters or scan some OMR sheets first.
                          </div>
                        )}
                      </div>

                     {/* Export Actions */}
<div className="flex flex-wrap gap-4 mt-6 justify-center">
  <button
    onClick={handleExportAllToCSV}
    className="bg-blue-500/20 hover:bg-blue-500/30 text-blue-300 px-6 py-3 rounded-lg transition-colors flex items-center space-x-2"
  >
    <Download size={20} />
    <span>Export All to CSV</span>
  </button>
  <button
    onClick={handleExportAllToExcel}
    className="bg-green-500/20 hover:bg-green-500/30 text-green-300 px-6 py-3 rounded-lg transition-colors flex items-center space-x-2"
  >
    <FileSpreadsheet size={20} />
    <span>Export All to Excel</span>
  </button>
  <button
    onClick={() => window.print()}
    className="bg-purple-500/20 hover:bg-purple-500/30 text-purple-300 px-6 py-3 rounded-lg transition-colors flex items-center space-x-2"
  >
    <FileText size={20} />
    <span>Print Report</span>
  </button>
</div>
                    </>
                  );
                })()}
              </CardContent>
            </Card>
          </motion.div>
        )}

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
                    {results.map((result) => (
                      <motion.div
                        key={result.id}
                        whileHover={{ scale: 1.02 }}
                        className="bg-white/5 rounded-lg p-6 border border-white/10 hover:border-white/20 transition-all duration-300 relative"
                      >
                        {/* Multiple answers warning badge */}
                        {result.multiple_answers > 0 && (
                          <div className="inline-block bg-yellow-500 text-black rounded-full px-2 py-1 text-xs font-bold mb-2">
                            {result.multiple_answers} Multiple
                          </div>
                        )}
                        
                        <div className="flex justify-between items-start mb-4">
                          <div className="flex-1">
                            <h4 className="font-semibold text-lg mb-1">
                              {result.student_name || 'Unknown Student'}
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
                            <span className="text-green-400">Correct: {result.correct_answers || result.score || 0}</span>
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
                            <AlertTriangle size={12} className="text-orange-400" />
                            <span className="text-orange-400 font-bold">Multiple: {result.multiple_answers || 0}</span>
                          </div>
                        </div>

                        {/* Answer Pattern Display */}
                        {result.detected_answers && (
                          <div className="mb-4">
                            <p className="text-xs text-gray-400 mb-2">Detected Answers:</p>
                            <div className="text-xs font-mono bg-black/30 p-3 rounded max-h-16 overflow-y-auto">
                              {typeof result.detected_answers === 'object' 
                                ? Object.entries(result.detected_answers).map(([q, ans]) => (
                                    <span key={q} className={ans === 'MULTIPLE' ? 'text-orange-400 font-bold' : ''}>
                                      Q{q}:{ans}{' '}
                                    </span>
                                  ))
                                : result.detected_answers}
                            </div>
                          </div>
                        )}

                        <div className="flex flex-col sm:flex-row gap-2">
                          <button
                            onClick={() => handleExportResult(result.id)}
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
                            onClick={() => handleDeleteResult(result.id)}
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

                {/* Warning if multiple answers detected */}
                {currentResult.multiple_answers > 0 && (
                  <div className="bg-yellow-500/20 border border-yellow-500/50 rounded-lg p-4 mb-6">
                    <div className="flex items-center space-x-2">
                      <AlertTriangle size={24} className="text-yellow-400" />
                      <div>
                        <h4 className="font-bold text-yellow-300">Multiple Bubbles Detected!</h4>
                        <p className="text-yellow-200 text-sm">
                          {currentResult.multiple_answers} question(s) have multiple bubbles marked. 
                          These are marked as "MULTIPLE" and counted as incorrect.
                        </p>
                      </div>
                    </div>
                  </div>
                )}

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
                          <CreditCard size={16} className="text-gray-400" />
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
                            <div className="text-2xl font-bold text-green-400">
                              {currentResult.correct_answers || currentResult.score || 0}
                            </div>
                            <div className="text-sm text-green-300">Correct</div>
                          </div>
                          <div className="text-center bg-red-500/10 rounded-lg p-4">
                            <div className="text-2xl font-bold text-red-400">
                              {currentResult.wrong_answers || 0}
                            </div>
                            <div className="text-sm text-red-300">Wrong</div>
                          </div>
                          <div className="text-center bg-yellow-500/10 rounded-lg p-4">
                            <div className="text-2xl font-bold text-yellow-400">
                              {currentResult.blank_answers || 0}
                            </div>
                            <div className="text-sm text-yellow-300">Blank</div>
                          </div>
                          <div className="text-center bg-orange-500/10 rounded-lg p-4">
                            <div className="text-2xl font-bold text-orange-400">
                              {currentResult.multiple_answers || 0}
                            </div>
                            <div className="text-sm text-orange-300">Multiple</div>
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
                                detail.detected === 'MULTIPLE' ? 'bg-orange-500/20 text-orange-300 border border-orange-500/30' :
                                detail.is_correct || detail.status === '✓' ? 'bg-green-500/10 text-green-300 border border-green-500/20' : 
                                detail.detected === 'N/A' || detail.detected === 'BLANK' || detail.status === '-' ? 'bg-yellow-500/10 text-yellow-300 border border-yellow-500/20' :
                                'bg-red-500/10 text-red-300 border border-red-500/20'
                              }`}
                            >
                              <span className="font-medium">Q{detail.question || index + 1}:</span>
                              <div className="flex items-center space-x-3">
                                <span className={detail.detected === 'MULTIPLE' ? 'font-bold text-orange-400' : ''}>
                                  <span className="text-gray-400">Detected:</span> {detail.detected || detail.detected_answer || 'N/A'}
                                </span>
                                <span>
                                  <span className="text-gray-400">Correct:</span> {detail.correct || detail.correct_answer || 'N/A'}
                                </span>
                                <span className="text-lg">
                                  {detail.detected === 'MULTIPLE' ? '⚠️' :
                                   detail.is_correct || detail.status === '✓' ? '✓' : 
                                   detail.detected === 'N/A' || detail.detected === 'BLANK' || detail.status === '-' ? '○' : '✗'}
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
                                    <span key={i} className={`inline-block mr-4 mb-2 ${ans === 'MULTIPLE' ? 'text-orange-400 font-bold' : ''}`}>
                                      Q{q}: <span className={ans === 'MULTIPLE' ? 'text-orange-400' : 'text-green-400'}>{ans}</span>
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
                      if (currentResult && currentResult.id) {
                        handleExportResult(currentResult.id);
                      } else if (currentResult && currentResult.student_id) {
                        handleExportResult(currentResult.student_id);
                      } else {
                        showNotification("No result ID found", "error");
                      }
                    }}
                    disabled={exportLoading || !serverAwake}
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
          <p className="mb-2">Enhanced OMR Scanner v7.0 - Multiple Bubble Detection Fixed</p>
          <p className="flex items-center justify-center space-x-2">
            {serverAwake ? (
              answerKeyLoaded ? (
                <><CheckCircle size={16} className="text-green-400" /><span>Ready to scan</span></>
              ) : (
                <><XCircle size={16} className="text-red-400" /><span>Load answer key to begin</span></>
              )
            ) : (
              <><AlertCircle size={16} className="text-yellow-400" /><span>Connecting to server...</span></>
            )}
          </p>
          {window.location.hostname !== 'localhost' && (
            <p className="mt-2 text-xs text-gray-500">
              Hosted on free tier - Server may take 30-50 seconds to wake up after inactivity
            </p>
          )}
        </div>
      </div>
    </div>
  );
}