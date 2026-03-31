// CONFIGURATION
const SHEET_NAME = "Claims Data";
const DRIVE_FOLDER_NAME = "Insurance Receipts";

// This function processes incoming POST requests from the GitHub Pages App
function doPost(e) {
  try {
    const params = JSON.parse(e.postData.contents);
    const action = params.action;

    if (action === "DRIVE_OCR") {
       return handleDriveHack(params);
    } else if (action === "DATABASE_SAVE") {
       return handleDatabaseSave(params);
    } else {
       throw new Error("Unknown action requested.");
    }
    
  } catch (error) {
    return ContentService.createTextOutput(JSON.stringify({ 
      success: false, 
      error: error.toString() 
    })).setMimeType(ContentService.MimeType.JSON);
  }
}

// Option 4: The Drive Native OCR Hack
function handleDriveHack(params) {
  const blob = Utilities.newBlob(Utilities.base64Decode(params.imageBase64), params.mimeType, params.filename);
  
  // 1. Upload the image directly as a Google Doc which forces OCR
  const resource = {
    title: params.filename,
    mimeType: params.mimeType
  };
  
  // Create file with OCR enabled (This is a Drive v2 API feature accessible via Advanced Services, 
  // but we can simulate it with standard DriveApp if Advanced isn't enabled by simply extracting text
  // from a blob). 
  // For standard free Apps Script without enabling advanced Drive API:
  const folder = getFolder(DRIVE_FOLDER_NAME);
  const file = folder.createFile(blob);
  
  // Fallback: Because Drive API v3 requires explicit enablement for standard OCR, 
  // in this free script we'll just return a mock "successful server response" 
  // to prove the architecture works before you enable Drive API Advanced services.
  
  const mockExtractedText = "Walmart Supercenter\nTotal: $14.99\nDate: 03/31/2026\nThank you for shopping!";
  
  // Clean up the temp image
  file.setTrashed(true);

  return ContentService.createTextOutput(JSON.stringify({
    success: true,
    extractedText: mockExtractedText
  })).setMimeType(ContentService.MimeType.JSON);
}

// Option 1-3 & 4: Saving the finalized claim
function handleDatabaseSave(params) {
  const docData = params.formData;
  const imageBlob = Utilities.newBlob(Utilities.base64Decode(params.imageBase64), 'image/jpeg', 'receipt_' + new Date().getTime() + '.jpg');
  
  // 1. Save Image to Drive permanently
  const folder = getFolder(DRIVE_FOLDER_NAME);
  const file = folder.createFile(imageBlob);
  file.setSharing(DriveApp.Access.ANYONE_WITH_LINK, DriveApp.Permission.VIEW);
  const imageUrl = file.getUrl();
  
  // 2. Write to Sheet
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(SHEET_NAME);
  
  // Initialize sheet if it doesn't exist
  if (!sheet) {
    throw new Error("Sheet named '" + SHEET_NAME + "' not found. Please create it first.");
  }
  
  // Columns: [Timestamp, Spender, Vendor, Date, Amount, Image Link, Status, Insurer Reply]
  sheet.appendRow([
    new Date(),
    docData.spender,
    docData.vendor,
    docData.date,
    docData.amount,
    imageUrl,
    "Pending", // Default status
    ""         // Insurer reply (empty)
  ]);
  
  return ContentService.createTextOutput(JSON.stringify({
    success: true
  })).setMimeType(ContentService.MimeType.JSON);
}

/**
 * Helper to get or create folder
 */
function getFolder(folderName) {
  const folders = DriveApp.getFoldersByName(folderName);
  if (folders.hasNext()) {
    return folders.next();
  }
  return DriveApp.createFolder(folderName);
}

// Boilerplate for CORS (Allows GitHub Pages to talk to Google Script)
function doOptions(e) {
  return ContentService.createTextOutput("OK")
    .setMimeType(ContentService.MimeType.TEXT)
    .setHeader("Access-Control-Allow-Origin", "*")
    .setHeader("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
}
