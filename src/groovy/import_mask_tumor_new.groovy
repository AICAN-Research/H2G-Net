import qupath.lib.images.servers.ImageServerProvider
import qupath.lib.regions.RegionRequest
import java.awt.image.BufferedImage

import static qupath.lib.gui.scripting.QPEx.*


// --- SET THESE PARAMETERS ---
def masksPath = "C:/path/to/fastpathology/projects/projectName/results/dir/"
def className = "Tumor"
def postProcess = true;  // whether to apply post-processing smoothing to annotation
// ----------------------


// Get a list of image files, stopping early if none can be found
def dirOutput = new File(masksPath);
if (!dirOutput.isDirectory()) {
    print dirOutput + ' is not a valid directory!';
    return;
}

// get current WSI name
def currWSIName = GeneralTools.getNameWithoutExtension(getProjectEntry().getImageName())

// attempt to fix name if cellSens VSI
if (currWSIName.contains(".vsi")) {
    currWSIName = currWSIName.split(".vsi")[0]
}
seg_filepath = "Breast Tumour Segmentation/segmentation/segmentation.tiff"
def currFullPath = masksPath + "/" + currWSIName + "/" + seg_filepath

// check if file exists, if no return
File file = new File(currFullPath)
if (!file.exists()) {
    print currFullPath + ' does not exist!';
    return;
}

// Ideally you'd use ImageIO.read(File)... but if it doesn't work we need this
// we load the segmentation image into QuPath as image
def server = ImageServerProvider.buildServer(currFullPath, BufferedImage)
def region = RegionRequest.createInstance(server)
def img = server.readBufferedImage(region)

// calculate spacing based on seg image size and corresponding WSI
def wsi_server = getCurrentServer()

def seg_height = server.getHeight()
def seg_width = server.getWidth()

def scale_height = wsi_server.getHeight() / server.getHeight()
def scale_width = wsi_server.getWidth() / server.getWidth()

// create annotation
def band = ContourTracing.extractBand(img.getRaster(), 0)
def request = RegionRequest.createInstance(getCurrentServer(), 1.0)
def annotations = ContourTracing.createAnnotations(band, request, 1, 1)
addObjects(annotations)

// resize annotation to right size
def new_annotations = getAnnotationObjects().findAll()
def roi = new_annotations[0].getROI()
def roiScaled = roi.scale(scale_width, scale_height, 0, 0)
new_annotations[0].setROI(roiScaled)

// delete old annotation and insert resized one
removeObjects(annotations, false)
addObjects(new_annotations)

// attempt to smooth edges through morphological post-processing
if (postProcess) {
    selectAnnotations();
    runPlugin('qupath.lib.plugins.objects.DilateAnnotationPlugin', '{"radiusMicrons": -50,  "lineCap": "Round",  "removeInterior": false,  "constrainToParent": false}');
    clearSelectedObjects(true);
    
    selectAnnotations();
    runPlugin('qupath.lib.plugins.objects.DilateAnnotationPlugin', '{"radiusMicrons": 100,  "lineCap": "Round",  "removeInterior": false,  "constrainToParent": false}');
    clearSelectedObjects(true);
    
    selectAnnotations();
    runPlugin('qupath.lib.plugins.objects.DilateAnnotationPlugin', '{"radiusMicrons": -50,  "lineCap": "Round",  "removeInterior": false,  "constrainToParent": false}');
    clearSelectedObjects(true);
 }

// finally, rename to class of interest
replaceClassification(null, className);
