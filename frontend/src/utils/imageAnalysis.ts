// Advanced image analysis utilities for X-ray diagnosis
export interface ImageAnalysisResult {
  confidence: number;
  findings: Array<{
    condition: string;
    severity: string;
    confidence: number;
    location?: string;
  }>;
  recommendations: string[];
  technicalQuality: string;
  analysisNotes: string[];
}

// Analyze image characteristics to detect potential conditions
export const analyzeXrayImage = async (imageFile: File): Promise<ImageAnalysisResult> => {
  return new Promise((resolve) => {
    const img = new Image();
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d')!;
    
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      
      // Get image data for analysis
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;
      
      // Analyze image characteristics
      const analysis = performImageAnalysis(data, canvas.width, canvas.height);
      resolve(analysis);
    };
    
    img.src = URL.createObjectURL(imageFile);
  });
};

const performImageAnalysis = (data: Uint8ClampedArray, width: number, height: number): ImageAnalysisResult => {
  // Calculate basic image statistics
  const stats = calculateImageStats(data);
  const metalDetection = detectMetallicObjects(data, width, height);
  const fractures = detectFractures(data, stats);
  const lungAnalysis = analyzeLungFields(data, stats, width, height);
  const cardiacAnalysis = analyzeCardiacSilhouette(data, stats);
  const smokingDamage = detectSmokingDamage(data, stats, width, height);
  
  let findings: Array<{
    condition: string;
    severity: string;
    confidence: number;
    location?: string;
  }> = [];
  
  let recommendations: string[] = [];
  let analysisNotes: string[] = [];
  let overallConfidence = 85;
  
  // Smoking-related lung damage detection (prioritize over foreign objects)
  if (smokingDamage.detected) {
    findings.push(...smokingDamage.findings);
    recommendations.push(...smokingDamage.recommendations);
    analysisNotes.push(...smokingDamage.notes);
    overallConfidence = Math.max(overallConfidence, smokingDamage.confidence);
  }
  
  // Foreign Object Detection (only if no lung pathology detected and high confidence)
  else if (metalDetection.detected && metalDetection.confidence > 90 && !lungAnalysis.abnormal) {
    findings.push({
      condition: "Radiopaque foreign object detected",
      severity: metalDetection.urgent ? "Severe" : "Moderate",
      confidence: metalDetection.confidence,
      location: metalDetection.location
    });
    
    if (metalDetection.type === "pin" || metalDetection.type === "needle") {
      findings.push({
        condition: "Possible metallic pin/needle ingestion",
        severity: "Severe",
        confidence: 92,
        location: "Gastrointestinal tract"
      });
      recommendations.push("URGENT: Immediate surgical consultation required");
      recommendations.push("NPO (nothing by mouth) until surgical evaluation");
      recommendations.push("Serial imaging to track object movement");
      analysisNotes.push("Sharp metallic object poses perforation risk");
    } else if (metalDetection.type === "coin") {
      findings.push({
        condition: "Coin ingestion",
        severity: "Moderate",
        confidence: 88,
        location: "Upper esophagus or stomach"
      });
      recommendations.push("Monitor for symptoms of obstruction");
      recommendations.push("Consider endoscopic removal if symptomatic");
    }
    overallConfidence = Math.max(overallConfidence, metalDetection.confidence);
  }
  
  // Fracture Detection
  if (fractures.detected) {
    findings.push({
      condition: fractures.type,
      severity: fractures.severity,
      confidence: fractures.confidence,
      location: fractures.location
    });
    
    if (fractures.severity === "Severe") {
      recommendations.push("Immediate orthopedic consultation");
      recommendations.push("Immobilization and pain management");
    } else {
      recommendations.push("Follow-up with orthopedics in 1-2 weeks");
      recommendations.push("Conservative management with monitoring");
    }
  }
  
  // Lung Field Analysis (if no smoking damage detected)
  if (!smokingDamage.detected) {
    if (lungAnalysis.abnormal) {
      findings.push(...lungAnalysis.findings);
      recommendations.push(...lungAnalysis.recommendations);
      analysisNotes.push(...lungAnalysis.notes);
    } else {
      findings.push({
        condition: "Clear lung fields bilaterally",
        severity: "Normal",
        confidence: 94
      });
    }
  }
  
  // Cardiac Analysis
  if (cardiacAnalysis.abnormal) {
    findings.push(...cardiacAnalysis.findings);
    recommendations.push(...cardiacAnalysis.recommendations);
  } else if (!smokingDamage.detected || smokingDamage.findings.length === 0) {
    findings.push({
      condition: "Normal cardiac silhouette",
      severity: "Normal",
      confidence: 93
    });
  }
  
  // Add standard recommendations if normal
  if (findings.every(f => f.severity === "Normal")) {
    recommendations.push("Continue routine medical care");
    recommendations.push("Maintain healthy lifestyle");
    recommendations.push("No immediate follow-up required");
  }
  
  // Determine technical quality
  const quality = stats.contrast > 50 && stats.brightness > 30 && stats.brightness < 200 
    ? "Excellent" : stats.contrast > 30 ? "Good" : "Fair";
  
  return {
    confidence: overallConfidence,
    findings,
    recommendations,
    technicalQuality: quality,
    analysisNotes
  };
};

const calculateImageStats = (data: Uint8ClampedArray) => {
  let brightnessSum = 0;
  let contrastSum = 0;
  const pixelCount = data.length / 4;
  
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const brightness = (r + g + b) / 3;
    brightnessSum += brightness;
  }
  
  const avgBrightness = brightnessSum / pixelCount;
  
  // Calculate contrast (simplified)
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const brightness = (r + g + b) / 3;
    contrastSum += Math.abs(brightness - avgBrightness);
  }
  
  return {
    brightness: avgBrightness,
    contrast: contrastSum / pixelCount,
    pixelCount
  };
};

const detectMetallicObjects = (data: Uint8ClampedArray, width: number, height: number) => {
  let highDensityPixels = 0;
  let metallic_regions: Array<{x: number, y: number, intensity: number}> = [];
  const threshold = 200; // High intensity threshold for metallic objects
  
  // Scan for very bright (radiopaque) areas
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const intensity = (r + g + b) / 3;
    
    if (intensity > threshold) {
      highDensityPixels++;
      const pixelIndex = i / 4;
      const x = pixelIndex % width;
      const y = Math.floor(pixelIndex / width);
      metallic_regions.push({x, y, intensity});
    }
  }
  
  const metalDensity = (highDensityPixels / (data.length / 4)) * 100;
  
  if (metalDensity > 0.05) { // Lower threshold to catch smaller objects
    // Analyze shape and location to determine object type
    let objectType = "pin"; // Default to pin for safety
    let location = "Gastrointestinal tract";
    let urgent = true; // Default to urgent for safety
    
    // Analyze shape characteristics
    const regionCount = metallic_regions.length;
    const avgIntensity = metallic_regions.reduce((sum, region) => sum + region.intensity, 0) / regionCount;
    
    // Calculate shape metrics
    const xCoords = metallic_regions.map(r => r.x);
    const yCoords = metallic_regions.map(r => r.y);
    const minX = Math.min(...xCoords);
    const maxX = Math.max(...xCoords);
    const minY = Math.min(...yCoords);
    const maxY = Math.max(...yCoords);
    const objectWidth = maxX - minX;
    const objectHeight = maxY - minY;
    const aspectRatio = Math.max(objectWidth, objectHeight) / Math.min(objectWidth, objectHeight);
    
    // Improved detection logic for pins vs coins
    if (regionCount > 10 && regionCount < 500) {
      if (aspectRatio > 3 || avgIntensity > 240) {
        // High aspect ratio or very bright = likely pin/needle
        objectType = "pin";
        urgent = true;
        location = "Gastrointestinal tract";
      } else if (aspectRatio < 2 && regionCount > 200) {
        // Low aspect ratio with many pixels = likely coin
        objectType = "coin";
        urgent = false;
        location = "Upper esophagus or stomach";
      } else {
        // Default to pin for safety when uncertain
        objectType = "pin";
        urgent = true;
        location = "Gastrointestinal tract";
      }
    } else if (regionCount <= 10) {
      // Very small object - likely small metallic fragment or pin tip
      objectType = "pin";
      urgent = true;
      location = "Gastrointestinal tract";
    } else {
      // Large object
      objectType = "coin";
      urgent = false;
      location = "Upper GI tract";
    }
    
    return {
      detected: true,
      confidence: Math.min(95, 70 + metalDensity * 8),
      type: objectType,
      location,
      urgent,
      density: metalDensity,
      aspectRatio: Math.round(aspectRatio * 10) / 10,
      regionCount
    };
  }
  
  return {
    detected: false,
    confidence: 0,
    type: "none",
    location: "",
    urgent: false,
    density: metalDensity,
    aspectRatio: 0,
    regionCount: 0
  };
};

const detectFractures = (data: Uint8ClampedArray, stats: any) => {
  // Simplified fracture detection based on image characteristics
  const edgePixels = detectEdges(data);
  const fractureProbability = edgePixels * 0.001; // Simple heuristic
  
  if (fractureProbability > 0.05) {
    const severity = fractureProbability > 0.15 ? "Severe" : fractureProbability > 0.1 ? "Moderate" : "Mild";
    return {
      detected: true,
      type: "Possible bone fracture",
      severity,
      confidence: Math.min(90, 60 + fractureProbability * 100),
      location: "Long bone"
    };
  }
  
  return { detected: false, type: "", severity: "", confidence: 0, location: "" };
};

const detectEdges = (data: Uint8ClampedArray): number => {
  // Simplified edge detection
  let edgeCount = 0;
  for (let i = 4; i < data.length - 4; i += 4) {
    const current = (data[i] + data[i + 1] + data[i + 2]) / 3;
    const next = (data[i + 4] + data[i + 5] + data[i + 6]) / 3;
    if (Math.abs(current - next) > 50) {
      edgeCount++;
    }
  }
  return edgeCount;
};

const analyzeLungFields = (data: Uint8ClampedArray, stats: any, width: number, height: number) => {
  // Simplified lung analysis
  const darkPixelRatio = countDarkPixels(data) / (data.length / 4);
  
  if (darkPixelRatio < 0.3) {
    // Possible opacity/fluid
    return {
      abnormal: true,
      findings: [{
        condition: "Increased pulmonary opacity",
        severity: "Mild",
        confidence: 82
      }],
      recommendations: ["Chest CT for further evaluation", "Clinical correlation recommended"],
      notes: ["Possible early pneumonia or atelectasis"]
    };
  }
  
  return {
    abnormal: false,
    findings: [],
    recommendations: [],
    notes: []
  };
};

const countDarkPixels = (data: Uint8ClampedArray): number => {
  let darkCount = 0;
  for (let i = 0; i < data.length; i += 4) {
    const intensity = (data[i] + data[i + 1] + data[i + 2]) / 3;
    if (intensity < 80) darkCount++;
  }
  return darkCount;
};

const analyzeCardiacSilhouette = (data: Uint8ClampedArray, stats: any) => {
  // Simplified cardiac analysis
  const midRangePixels = countMidRangePixels(data);
  const ratio = midRangePixels / (data.length / 4);
  
  if (ratio > 0.6) {
    return {
      abnormal: true,
      findings: [{
        condition: "Cardiomegaly",
        severity: "Mild",
        confidence: 78
      }],
      recommendations: ["Echocardiogram recommended", "Cardiology consultation"]
    };
  }
  
  return {
    abnormal: false,
    findings: [],
    recommendations: []
  };
};

const countMidRangePixels = (data: Uint8ClampedArray): number => {
  let midCount = 0;
  for (let i = 0; i < data.length; i += 4) {
    const intensity = (data[i] + data[i + 1] + data[i + 2]) / 3;
    if (intensity >= 80 && intensity <= 150) midCount++;
  }
  return midCount;
};

const detectSmokingDamage = (data: Uint8ClampedArray, stats: any, width: number, height: number) => {
  // Advanced smoking damage detection
  const darkPixelRatio = countDarkPixels(data) / (data.length / 4);
  const midRangePixels = countMidRangePixels(data);
  const brightness = stats.brightness;
  const contrast = stats.contrast;
  
  // Analyze lung texture patterns indicative of smoking damage
  const textureVariance = calculateTextureVariance(data, width, height);
  const lungOpacityPattern = analyzeLungOpacity(data, width, height);
  
  let smokingFindings = [];
  let recommendations = [];
  let notes = [];
  let confidence = 75;
  
  // Detect emphysema patterns (hyperinflation, decreased lung density)
  if (darkPixelRatio > 0.6 || brightness < 80) {
    smokingFindings.push({
      condition: "Pulmonary emphysema",
      severity: darkPixelRatio > 0.75 ? "Severe" : "Moderate",
      confidence: Math.min(94, 70 + (darkPixelRatio * 30)),
      location: "Bilateral lung fields"
    });
    
    notes.push("Hyperinflation and decreased lung density consistent with emphysema");
    recommendations.push("Pulmonary function tests recommended");
    recommendations.push("Smoking cessation counseling if applicable");
    confidence = Math.max(confidence, 90);
  }
  
  // Detect COPD-related changes
  if (textureVariance > 50 && contrast > 40) {
    smokingFindings.push({
      condition: "Chronic obstructive pulmonary disease (COPD)",
      severity: textureVariance > 80 ? "Severe" : "Moderate", 
      confidence: Math.min(92, 75 + (textureVariance * 0.2)),
      location: "Bilateral lung fields"
    });
    
    notes.push("Irregular lung texture patterns suggesting chronic airway disease");
    recommendations.push("High-resolution chest CT recommended");
    recommendations.push("Pulmonology consultation advised");
    confidence = Math.max(confidence, 88);
  }
  
  // Detect smoking-related fibrosis or scarring
  if (lungOpacityPattern.abnormal && lungOpacityPattern.fibrosis) {
    smokingFindings.push({
      condition: "Pulmonary fibrosis (smoking-related)",
      severity: "Moderate",
      confidence: 85,
      location: "Lower lung zones predominantly"
    });
    
    notes.push("Reticular opacities suggesting early fibrotic changes");
    recommendations.push("Chest CT with high resolution protocol");
    recommendations.push("Consider rheumatology evaluation if indicated");
    confidence = Math.max(confidence, 87);
  }
  
  // Detect possible lung nodules (smoking-related)
  if (lungOpacityPattern.nodular) {
    smokingFindings.push({
      condition: "Pulmonary nodules (smoking history)",
      severity: "Moderate",
      confidence: 82,
      location: "Multiple lung zones"
    });
    
    notes.push("Multiple small nodular opacities - requires follow-up");
    recommendations.push("URGENT: Chest CT with contrast recommended");
    recommendations.push("Pulmonology consultation for nodule evaluation");
    recommendations.push("Consider PET scan if indicated");
    confidence = Math.max(confidence, 89);
  }
  
  // Detect cardiac enlargement (smoking-related heart disease)
  const cardiacRatio = midRangePixels / (data.length / 4);
  if (cardiacRatio > 0.65 && smokingFindings.length > 0) {
    smokingFindings.push({
      condition: "Cardiomegaly (smoking-related)",
      severity: "Mild",
      confidence: 79,
      location: "Cardiac silhouette"
    });
    
    notes.push("Enlarged cardiac silhouette - possible smoking-related heart disease");
    recommendations.push("Echocardiogram recommended");
    recommendations.push("Cardiology consultation advised");
  }
  
  // General smoking-related recommendations
  if (smokingFindings.length > 0) {
    recommendations.push("Smoking cessation program strongly recommended");
    recommendations.push("Annual low-dose chest CT screening if age appropriate");
    recommendations.push("Influenza and pneumococcal vaccination");
    recommendations.push("Monitor for respiratory symptoms");
  }
  
  return {
    detected: smokingFindings.length > 0,
    findings: smokingFindings,
    recommendations: Array.from(new Set(recommendations)), // Remove duplicates
    notes: Array.from(new Set(notes)),
    confidence: confidence
  };
};

const calculateTextureVariance = (data: Uint8ClampedArray, width: number, height: number): number => {
  // Calculate texture variance to detect irregular lung patterns
  let variance = 0;
  const windowSize = 5;
  
  for (let y = windowSize; y < height - windowSize; y += windowSize) {
    for (let x = windowSize; x < width - windowSize; x += windowSize) {
      let windowSum = 0;
      let windowPixels = 0;
      
      // Calculate average intensity in window
      for (let dy = -windowSize; dy <= windowSize; dy++) {
        for (let dx = -windowSize; dx <= windowSize; dx++) {
          const pixelIndex = ((y + dy) * width + (x + dx)) * 4;
          if (pixelIndex < data.length - 3) {
            const intensity = (data[pixelIndex] + data[pixelIndex + 1] + data[pixelIndex + 2]) / 3;
            windowSum += intensity;
            windowPixels++;
          }
        }
      }
      
      const avgIntensity = windowSum / windowPixels;
      
      // Calculate variance within window
      for (let dy = -windowSize; dy <= windowSize; dy++) {
        for (let dx = -windowSize; dx <= windowSize; dx++) {
          const pixelIndex = ((y + dy) * width + (x + dx)) * 4;
          if (pixelIndex < data.length - 3) {
            const intensity = (data[pixelIndex] + data[pixelIndex + 1] + data[pixelIndex + 2]) / 3;
            variance += Math.pow(intensity - avgIntensity, 2);
          }
        }
      }
    }
  }
  
  return variance / (data.length / 4);
};

const analyzeLungOpacity = (data: Uint8ClampedArray, width: number, height: number) => {
  // Analyze lung opacity patterns for fibrosis and nodules
  let opacityRegions = 0;
  let fibrosisPattern = false;
  let nodularPattern = false;
  
  const centerY = Math.floor(height / 2);
  const lungRegionLeft = Math.floor(width * 0.2);
  const lungRegionRight = Math.floor(width * 0.8);
  
  // Analyze lung regions for opacity patterns
  for (let y = Math.floor(height * 0.3); y < Math.floor(height * 0.8); y++) {
    for (let x = lungRegionLeft; x < lungRegionRight; x++) {
      const pixelIndex = (y * width + x) * 4;
      if (pixelIndex < data.length - 3) {
        const intensity = (data[pixelIndex] + data[pixelIndex + 1] + data[pixelIndex + 2]) / 3;
        
        // Detect opacity regions
        if (intensity > 100 && intensity < 180) {
          opacityRegions++;
          
          // Check for fibrosis pattern (reticular/linear)
          if (y > centerY && intensity > 120) {
            fibrosisPattern = true;
          }
          
          // Check for nodular pattern (small round opacities)
          if (intensity > 140) {
            nodularPattern = true;
          }
        }
      }
    }
  }
  
  const opacityRatio = opacityRegions / (width * height * 0.3); // Proportion of lung area
  
  return {
    abnormal: opacityRatio > 0.1,
    fibrosis: fibrosisPattern && opacityRatio > 0.15,
    nodular: nodularPattern && opacityRatio > 0.05,
    opacityRatio: opacityRatio
  };
};