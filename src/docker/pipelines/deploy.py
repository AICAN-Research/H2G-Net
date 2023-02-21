import fast
import os


fast.Reporter.setGlobalReportMethod(fast.Reporter.COUT)


pipeline = fast.Pipeline('/opt/pipelines/tissue.fpl', {'wsi': '/opt/pipelines/A05.svs', 'output': '/opt/pipelines/prediction.png'})
pipeline.parse()
pipeline.getProcessObject('exporter').run()
print("Were export successful:", os.path.exists('/opt/pipelines/prediction.png'))

