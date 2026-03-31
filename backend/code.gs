// CONFIGURATION
const SHEET_NAME = "Claims Data";
const DRIVE_FOLDER_NAME = "Insurance Receipts";

// This function processes incoming POST requests from the GitHub Pages App
function doPost(e) {
  try {
    const params = JSON.parse(e.postData.contents);
    const action = params.action;

    if (action === "DATABASE_SAVE") {
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
