## the files :

**lightningboltProcessor.py** core function independant from Maya  
**mayaLightningPlugin.py** the plugin that declare the Maya node and send the proper informations to the processor in lightningboltProcessor.py, this is the file you will load as a plugin or see in the plug-in manager.  
**AElightningProcessorTemplate.mel** AE template script to have a nice display in the Attribute Editor.

## installing the files:

- copy the file **AElightningProcessorTemplate.mel** into the AETemplate folder

- you need to keep **lightningboltProcessor.py** and **mayaLightningPlugin.py** together.  
The best is to copy both of them in the Maya python plugin folder (something like *Documents\maya\\{mayaVersion}\plug-ins* or some of the folder that are defined in the environment variable `MAYA_PLUG_IN_PATH` ). You can now load the plugin inside Maya with the plug-in Manager.
But you can also put them anywhere and load the plugin using commands like this:
		
		import pymel.core as pm
		pm.loadPlugin('[PATH]/mayaLightningPlugin.py') # Replace [PATH] with the folder where you put the files

## dependencies

- **Pymel**  
it's better if pymel is installed and working inside Maya's Python. It's only important for the setup python script to create and connect the node to curves, it's actually not requiered by the Plugin. To test if you have pymel just execute this in the Python Script Editor of Maya:

		import pymel.core as pm

- **Numpy**  
you absolutely need to have Numpy installed in your Maya's python. This will work only for Maya 2011 and 2012, Maya 2013's python changed too much and it's a more complicated case (google "numpy for maya 2013" to see all the problems)  
My way to make numpy work is not too complicated, we are going to force Maya to load python modules installed in the Windows's python.  
Of course you should have a normal python installation on your computer, the python you can access directly in Windows with the command line. If you don't have any, then install a Python 2.6 since this is the Python version used by Maya. Then you can check if numpy is already installed in your Windows's Python and working by executing:  

		import numpy as np
		print np.arange(1,6)  
If there is an error, well, you need to install Numpy. You can download numpy installation from here [http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy), download this one `numpy-MKL-1.7.1.win-amd64-py2.6.‌exe` and install it.  
Once it is installed, test numpy again using in the Windows command line. Ok so if it works now we have to copy that to Maya's Python. you have to find where is your freshly installed numpy folder, to know it you can execute these commands in the windows command line :  

		import numpy as np
		print np.__file__  
It will output something like this:  
*`{YOUR_PYTHON_LIBRARY_PATH}`*`/numpy/__init__.pyc`  
Now let’s make it work with Maya. Create a file **userSetup.py** (or edit it if you already have one) in your Maya scripts folder (for windows it’s something like `...\My Documents\maya\versionofmaya\scripts`). Open it and add this:

		import os
		import sys
		sys.path.append('{YOUR_PYTHON_LIBRARY_PATH}') # Put the result of what you got with the command before  
Save it and that should be good, you can start Maya and test numpy inside Maya in the Python Script Editor with:  

		import numpy as np
		print np.arange(1,6)  
there should be no error! then you are ok and you can load the plugin

## Maya Usage

- load the plugin. (use the plug-in manager or use the command loadplugin)
- create one or more curves
- execute this script in the Python script editor:

```python
		import pymel.core as pm

		def getProcessorFromMesh( mesh ):
			inConns = pm.listConnections( 'polySurfaceShape1.inMesh', type='lightningProcessor', s=True, d=False, p=False )
			if inConns is not None and len(inConns)>0:
				return inConns[0]

		def curvesToLightning( elements ):
			lightningProcessorNode = None
			curveShapes = []

			# parsing inputs
			for e in elements:
				if pm.nodeType(e)== 'transform':
					childs = pm.listRelatives(e, s=True)
					if len(childs)>0 and pm.nodeType(childs[0]) == 'nurbsCurve' :
						curveShapes.append(childs[0])
					elif len(childs)>0 and pm.nodeType(childs[0]) == 'mesh' and lightningProcessorNode is None:
						lightningProcessorNode = getProcessorFromMesh(childs[0])
				elif pm.nodeType(e) == 'nurbsCurve' :
					curveShapes.append(e)
				elif pm.nodeType(e) == 'mesh' and lightningProcessorNode is None:
					lightningProcessorNode = getProcessorFromMesh(e)

			if len(curveShapes) < 1:
				return

			if lightningProcessorNode is None:
				# create processor and mesh node
				lightningProcessorNode = pm.createNode('lightningProcessor')
				meshNode = pm.createNode('mesh')
				pm.connectAttr( 'time1.outTime', lightningProcessorNode+'.time')
				pm.connectAttr( lightningProcessorNode+'.outputMesh', meshNode+'.inMesh')
				
				# create shader and apply to it
				mrVert = pm.createNode('mentalrayVertexColors')
				sh = pm.shadingNode( 'surfaceShader', asShader=True )
				set = pm.sets( renderable=True, noSurfaceShader=True, empty=True)
				pm.connectAttr( sh+'.outColor', set+'.surfaceShader', f=True)
				pm.sets( set, e=True, forceElement=meshNode )
				forceEval = pm.polyEvaluate( meshNode, v=True )
				pm.connectAttr( (meshNode+".colorSet[0].colorName") , (mrVert+".cpvSets[0]") )
				pm.connectAttr( (mrVert+'.outColor'), (sh+".outColor") )

			for cs in curveShapes:
				pm.connectAttr( cs+'.worldSpace[0]', lightningProcessorNode+'.inputCurves', nextAvailable=True)

			return lightningProcessorNode

		curvesToLightning( pm.ls(sl=True) ) # call with the current selection  
```
this script main command is:

		curvesToLightning( list )  
list can be a list of curve and the command will create a the lightning mesh for all the curves and a shader on that mesh to display the vertexColors.
list can also be one or more curve **and** an already created lightning mesh, in this case the curves are added to already created system.

