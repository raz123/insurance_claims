# Google Apps Script Setup Guide

Follow these 5 simple steps to connect your GitHub Pages Web App to your bare Google Spreadsheet database. No Google Cloud billing is required.

## 1. Prepare your Spreadsheet
1. Create a brand new Google Spreadsheet.
2. At the bottom tab, rename "Sheet1" to exactly **`Claims Data`** (case-sensitive).
3. Type the following headers into Row 1 (A through H):
   - A1: `Timestamp`
   - B1: `Spender`
   - C1: `Vendor`
   - D1: `Date`
   - E1: `Amount`
   - F1: `Image Link`
   - G1: `Status`
   - H1: `Insurer Reply`

## 2. Open Apps Script
1. In your Spreadsheet menu, click **Extensions > Apps Script**.
2. A new tab will open with a file called `Code.gs`.
3. Delete the empty `myFunction() {}` block.

## 3. Paste the Code
1. Open the `code.gs` file from this repository (`backend/code.gs`).
2. Copy the entire contents of that file and paste it into the Google Apps Script editor.
3. Click the floppy disk icon (💾) at the top to save your project. Name the project "Insurance Claims API".

## 4. Deploy the Web App
1. In the top right corner of the Apps Script editor, click the blue **Deploy** button.
2. Select **New deployment**.
3. Click the gear icon (⚙️) next to "Select type" and choose **Web app**.
4. Fill out the configuration:
   - **Description:** `V1 Launch`
   - **Execute as:** `Me (your@email.com)`
   - **Who has access:** `Anyone` *(Crucial: This must be "Anyone", NOT "Anyone with Google Account", otherwise the GitHub Page cannot send data to it).*
5. Click **Deploy**.
6. Google will warn you that this app isn't verified.
   - Click **Review Permissions**.
   - Choose your Google Account.
   - Click **Advanced** at the bottom.
   - Click the unsafe link at the very bottom: `Go to Insurance Claims API (unsafe)`.
   - Click **Allow**.

## 5. Copy your API URL
1. You will be given a long **Web app URL** that ends in `/exec`.
2. Copy this URL!
3. Go to your new GitHub Pages web app, click the Settings Gear ⚙️ in the top right, and paste this URL into the input field.

**You are done! Your 100% free serverless architecture is perfectly linked.**
