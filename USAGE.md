# FS FilterLab - Usage Guide

This guide walks you through using FS FilterLab for optical analysis.

## Getting Started

### 1. Launch the Application
- **Windows**: Double-click `start.bat`
- **Manual**: Run `python program/app.py`
- The app will open automatically at [http://127.0.0.1:8080](http://127.0.0.1:8080)

### 2. Basic Interface Overview
- **Left Sidebar (Drawer)**: Filter selection, QE/illuminant pickers, settings, and actions
- **Main Content Area**: Interactive Plotly charts, metrics tables, reflector list
- **Notifications**: Success/error toasts appear in the top-right corner

## Basic Workflow

### Step 1: Select Filters
1. In the sidebar, use the **"Select filters to plot"** dropdown
2. Choose one or more filters from the list
3. Selected filters appear as colored tags
4. Use the **search box** to quickly find specific filters

### Step 2: Configure Filter Stack
1. If multiple filters are selected, they automatically stack (multiply)
2. Use **"Set Filter Stack Counts"** to specify how many of each filter
3. Example: 2x UV Filter + 1x Polarizer = specific transmission curve

### Step 3: Choose Analysis Parameters
1. **Camera QE**: Select quantum efficiency profile (or use Default)
2. **Illuminant**: Choose light source (AM1.5 Global is typical daylight)
3. **Display Options**: Toggle RGB channels, log scale, white balance

### Step 4: Analyze Results
1. **Main transmission chart** shows the combined filter response
2. **RGB sensor response** shows how the filtered light affects camera channels
3. **Metrics panels** show quantitative analysis (effective stops, white balance)
4. **Deviation metrics** compare to target profiles (if loaded)

## Advanced Features

### Advanced Filter Search
1. Click **"Show Advanced Search"** in the sidebar
2. **Filter by Manufacturer**: Select brands to narrow choices
3. **Transmission at Wavelength**: Find filters with specific properties at target wavelengths
4. **Color Sorting**: Sort by rainbow color for visual selection

### Custom Data Import
1. Click **"Show Import Data"** in the sidebar
2. **Upload Filter Data**: Import custom TSV filter files
3. **Upload QE Data**: Add camera sensor profiles
4. **Upload Illuminant**: Add custom light sources
5. **Upload Reflectance**: Add surface material data

### Report Generation
1. Select your desired filter configuration
2. Choose the camera profile for analysis
3. Click **"Generate Report"** in the sidebar
4. Download the generated PNG report with **"Download Last Report"**

## Understanding the Charts

### Transmission Chart
- **X-axis**: Wavelength (300-1100 nm)
- **Y-axis**: Transmission (0-100% or logarithmic)
- **Multiple Lines**: Individual filters in stack
- **Combined Line**: Final result of all filters

### RGB Response Chart
- **Red/Green/Blue Lines**: How each camera channel responds
- **Combined Effect**: Shows color shifts and intensity changes
- **White Balance**: Correction factors for neutral response

### Sparkline Plots
- **Miniature Charts**: Quick visual summary in selection lists
- **Filter Overview**: Rapid identification of filter characteristics
- **Comparison Tool**: Easy visual comparison between options


## Settings and Preferences

### Display Options
- **Log View**: Toggle between linear and logarithmic transmission scales
- **RGB Channels**: Show/hide individual red, green, blue responses
- **Apply White Balance**: Enable automatic white balance correction

### Performance Options
- **Rebuild Cache**: Clear cached data when adding new files
- **Advanced Search**: Enable multi-criteria filter search
- **Import Data**: Enable file upload interfaces
