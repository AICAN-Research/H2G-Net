guiscript=true
import qupath.lib.gui.QuPathGUI;

// in order to run script from command line
import static qupath.lib.scripting.QP.*
import qupath.lib.scripting.QP
import qupath.lib.gui.QuPathGUI

import static qupath.lib.roi.PathROIToolsAwt.getShape;
import java.awt.image.BufferedImage
import java.awt.Color
import javax.imageio.ImageIO
import qupath.lib.io.PathIO

import qupath.lib.scripting.QPEx // <- works for < Java 11 (pre QuPath-0.2.0)
//import qupath.lib.gui.scripting.QPEx // <- for Java 11 (QuPath-0.2.0)

def qupath = QPEx.getQuPath()

// set downsampling level
double downsample = 4

// Output directory for storing the tiles
def pathOutput = QPEx.buildFilePath(QPEx.PROJECT_BASE_DIR, 'exported_mask_png' + "_" + ((int)downsample).toString())
QPEx.mkdirs(pathOutput)

// Get the main QuPath data structures
def imageData = QPEx.getCurrentImageData()

def hierarchy = imageData.getHierarchy()
def servers = imageData.getServer()

String path = servers.getPath()
String name = path.tokenize('/')[-1]
String names = name.tokenize('.')[-2]

// only extract tumor class
def tumor = getPathClass('Tumor')
def annotations = getAnnotationObjects()
def tumorAnnotations = annotations.findAll {it.getPathClass() == tumor}
def shapes = tumorAnnotations.collect {getShape(it.getROI())}

// Create a grayscale image, here it's 10% of the full image size
def server = getCurrentImageData().getServer()
int w = (server.getWidth() / downsample) as int
int h = (server.getHeight() / downsample) as int
def img = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY)

// Paint the shapes (this is just 'standard' Java - you might want to modify)
def g2d = img.createGraphics()
g2d.scale(1.0/downsample, 1.0/downsample)
g2d.setColor(Color.WHITE)
for (shape in shapes)
    g2d.fill(shape)
g2d.dispose()

// Save the result -> save mask as PNG
def outputFile = new File(pathOutput, names + '.png') //getQuPath().getDialogHelper().promptToSaveFile("Save binary image", null, null, "PNG", ".png")
ImageIO.write(img, 'PNG', outputFile)
print 'Done!'
