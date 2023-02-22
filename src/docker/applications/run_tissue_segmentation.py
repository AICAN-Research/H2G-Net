import fast
import os


fast.Reporter.setGlobalReportMethod(fast.Reporter.COUT)

output = '/opt/pipelines/prediction.png'

pipeline = fast.Pipeline(
	'/opt/pipelines/tissue_segmentation.fpl',
	{
		'wsi': '/opt/pipelines/A05.svs',
		'output': output
	}
)

pipeline.parse()
pipeline.getProcessObject('exporter').run()

print("Was export successful:", os.path.exists(output))
print("Result is saved at:", output)
