#!/usr/bin/python
# -*- coding: iso-8859-1 -*-


'''

import pymel.core as pm

pm.loadPlugin('/Users/mathieu/Documents/IMPORTANT/Perso/Prog/python/lightningPlugin/mayaLightningPlugin.py')
lightNode = pm.createNode('lightningbolt')
meshNode = pm.createNode('mesh')
pm.connectAttr( lightNode+'.outputMesh', meshNode+'.inMesh')
pm.connectAttr( 'curveShape1.local', lightNode+'.inputCurve')

'''

import sys
import maya.OpenMaya as OpenMaya
import maya.OpenMayaMPx as OpenMayaMPx
import maya.mel as mel


kPluginNodeName = "lightningbolt"
kPluginNodeId = OpenMaya.MTypeId(0x8700B)

class attributeAccess:
	def __init__(self, mObject, index, isInput):
		self.mObject = mObject
		self.index = index
		self.isInput = isInput

class attCreationHlp:
	def __init__ (self,nodeClass):
		self.listAttIN = []
		self.listAttOUT = []
		self.nodeClass = nodeClass

	def __exit__(self, type, value, traceback):
		for attInTup in self.listAttIN :
			nameIn, mObjectIn, doAddIn, typeIn, exceptAffectList = attInTup
			delattr(self.nodeClass, nameIn )

		for attOutTup in self.listAttOUT :
			nameOut, mObjectOut, doAddOut, typeOut, exceptAffectList = attOutTup
			delattr(self.nodeClass, nameOut )


	def createAtt(self, name, fn, shortName, type, isInput=True, default=None, addAttr=True, childs = None, exceptAffectList = None ):
		sys.stderr.write('create '+name+' '+str(type)+'\n')
		tup = None
		list = self.listAttOUT
		if isInput:
			list = self.listAttIN

		if fn is OpenMaya.MRampAttribute :
			tup = ( name, fn.createCurveRamp(name, shortName) , addAttr, type )
		else :
			if childs is None :
				if default is not None:
					tup = ( name, fn.create(name,shortName, type, default ) , addAttr, type )
				else:
					tup = ( name, fn.create(name,shortName, type ) , addAttr, type )
			else:
				raise 'rework the child system'
				'''
				listAttFromChilds = []
				for c in childs :
					attTuple = list[c]
					if attTuple is not None:
						attFound,doAdd,t = attTuple
						listAttFromChilds.append(attFound)
				if len(childs) == 3 :
					list[name] = ( fn.create(name,shortName, listAttFromChilds[0], listAttFromChilds[1],listAttFromChilds[2] ) , addAttr, type  )
				'''
				
			if isInput == True :
				fn.setKeyable(True)
				fn.setWritable(True)
			else :	
				fn.setKeyable(False)
				fn.setWritable(False)
		
		if isInput == True :
			tup = tup + (exceptAffectList,)

		# we add the object to the class
		setattr(self.nodeClass, name, attributeAccess(tup[1], len(list),isInput) )
		list.append( tup )
	
	def addAllAttrs(self):
		for attInTup in self.listAttIN :
			name, mObject, doAdd, type, affectList = attInTup 
			if not doAdd:
				continue
			self.nodeClass.addAttribute( mObject )

		for attOutTup in self.listAttOUT :
			name, mObject, doAdd, type  = attOutTup 
			if not doAdd:
				continue
			self.nodeClass.addAttribute( mObject )

	def generateAffects (self) :
		for attOutTup in self.listAttOUT :
			nameOut, mObjectOut, doAddOut, typeOut  = attOutTup 
			for attInTup in self.listAttIN :
				nameIn, mObjectIn, doAddIn, typeIn, exceptAffectList = attInTup 
				if exceptAffectList is not None and nameOut in exceptAffectList :
					continue
				sys.stderr.write(nameIn+" affect "+nameOut+"\n")
				self.nodeClass.attributeAffects( mObjectIn, mObjectOut )
				#self.nodeClass.attributeAffects( getattr(self.nodeClass, nameIn), getattr(self.nodeClass, nameOut) )
	
	def getAttValueOrHdl( self, attribAccess, thisNode, data ):
		#sys.stderr.write("list IN "+str(lightningBoltNode.listAttIN)+"\n")
		#sys.stderr.write("list OUT "+str(lightningBoltNode.listAttOUT)+"\n")
		att = attribAccess.mObject
		attInTup = None
		if attribAccess.isInput:
			attInTup = self.listAttIN[attribAccess.index]
		else: 
			attInTup = self.listAttOUT[attribAccess.index]

		if not attribAccess.isInput:
			return data.outputValue( att )

		name, att2, doAdd, type, exceptAffectList = attInTup

		if type is "Ramp" :
			RampPlug = OpenMaya.MPlug( thisNode, att )
			RampPlug.setAttribute(att)
			RampPlug01 = OpenMaya.MPlug(RampPlug.node(), RampPlug.attribute())
			return OpenMaya.MRampAttribute(RampPlug01.node(), RampPlug01.attribute())
		
		tempData = data.inputValue(att)

		if type is OpenMaya.MFnNumericData.kDouble :
			return tempData.asDouble()
		if type is OpenMaya.MFnUnitAttribute.kTime :
			return tempData.asTime()
		if type is OpenMaya.MFnNumericData.kInt :
			return tempData.asInt()
		if type is OpenMaya.MFnData.kNurbsCurve :
			return tempData.asNurbsCurveTransformed()
		if type is "vectorOfDouble" :
			return OpenMaya.MVector( tempData.asVector() )


import numpy as np
import math

def loadStartBranchFrame( startDir ):
	startUp = np.array([0,1,0])
	startFront = startDir

	if abs( np.dot( startUp, startFront ) )>0.99999: # start of branch is parallel to Up
		startUp = np.array([1,0,0])

	startSide = np.cross( startFront, startUp)
	startSide /= math.sqrt( (startSide**2).sum(-1) )
	startUp = np.cross( startSide, startFront)

	return startFront, startUp, startSide 

def mayaLightningMesher(resultLoopingPoints, numLoopingLightningBranch, resultPoints, numLightningBranch, branchMaxPoints, tubeSide):
	
	facesConnect = [ ((id/2+1)%2 * ( ( id%4 + id/4 )%tubeSide) + (id/2)%2 *   (( (( 3-id%4 + id/4 )  )%tubeSide) + tubeSide)) + ring*tubeSide + branchNum*tubeSide*branchMaxPoints for branchNum in range(numLightningBranch) for ring in range(branchMaxPoints-1) for id in range(tubeSide*4) ]
	
	facesCount = [4]*(tubeSide*(branchMaxPoints-1)*numLightningBranch)

	numVertices = len(resultPoints)/4
	numFaces = len(facesCount)

	scrUtil = OpenMaya.MScriptUtil()
	scrUtil.createFromList( resultPoints, len(resultPoints))
	Mpoints = OpenMaya.MFloatPointArray( scrUtil.asDouble4Ptr(), numVertices)

	scrUtil.createFromList(facesConnect, len(facesConnect))
	MfaceConnects = OpenMaya.MIntArray( scrUtil.asIntPtr() , len(facesConnect))

	scrUtil.createFromList(facesCount, len(facesCount))
	MfacesCount = OpenMaya.MIntArray( scrUtil.asIntPtr(), numFaces)
	
	dataCreator = OpenMaya.MFnMeshData()
	outData = dataCreator.create()
	meshFS = OpenMaya.MFnMesh()


	meshFS.create(numVertices, numFaces, Mpoints , MfacesCount, MfaceConnects, outData)

	return outData


def mayaLightningMesherOLD( branchList ):
	tubeSide = 4
	pointsOfCircle = np.zeros( (4,tubeSide), np.float32 )
	pointsOfCircle[1] = np.sin(np.linspace(1.5*np.pi,-np.pi*.5, tubeSide, endpoint=False))
	pointsOfCircle[2] = np.cos(np.linspace(1.5*np.pi,-np.pi*.5, tubeSide, endpoint=False))
	pointsOfCircle[3] = np.ones(tubeSide)
	pointsOfCircle = pointsOfCircle.T
	#pointsOfCircle = np.array( [[ 0,  1,  0,  1],[0,  0,  1,  1],[ 0, -1,  0,  1],[ 0,  0,  -1,  1]] )

	points = []
	facesConnect =[]
	facesCount = []

	for b in branchList :
		path, attributes = b
		#sys.stderr.write("attributes "+str(attributes)+"\n")

		vectors = (np.roll(path,-3) - path)[:-1]
		normSqr = (vectors**2).sum(-1)
		vectorsUnit = vectors/np.sqrt(normSqr).reshape(-1,1)

		numPathPoints = len(path)
		#test = ((np.tile(pointsOfCircle,(numPathPoints,1,1))).reshape(numPathPoints*tubeSide,4)[:,:-1]).reshape(numPathPoints,tubeSide,3)
		#sys.stderr.write("circles "+str(test)+"\n")
		circles = np.tile(pointsOfCircle,(numPathPoints,1,1)) # make enough circles
		#sys.stderr.write("circles "+str(circles)+"\n")

		branchFrames = np.zeros( (numPathPoints,4,4), np.float32 )

		frame = np.array( [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]], np.float32 )
		prevFrame = np.array( [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]], np.float32 )
		branchFaceConnect = [ ((id/2+1)%2 * ( ( id%4 + id/4 )%tubeSide) + (id/2)%2 *   (( (( 3-id%4 + id/4 )  )%tubeSide) + tubeSide)) + ring*tubeSide for ring in range(numPathPoints-1) for id in range(tubeSide*4) ]
		branchFaceCount = [4]*(tubeSide*(numPathPoints-1))
		
		facesConnect.extend(branchFaceConnect)		
		facesCount.extend(branchFaceCount) 


		tpFr, tpUp, tpSi = loadStartBranchFrame( vectorsUnit[0] )

		prevFrame[0][:-1] = tpFr
		prevFrame[1][:-1] = tpUp
		prevFrame[2][:-1] = tpSi
		prevFrame[3][:-1] = path[0] # point of seedPath
		branchFrames[0] = prevFrame
		branchFrames[0][:-1,:-1]*=attributes[0][lightningBoltNode.LP_kRadius]

		for i in range(1,numPathPoints):
			if i<(numPathPoints-1):
				frame[0][:-1] = vectorsUnit[i-1] + vectorsUnit[i]
				frame[0] /= math.sqrt( (frame[0]**2).sum(-1) ) # normalize
			else:
				frame[0][:-1] = vectorsUnit[i-1]

			# calculate frame from prevFrame
			upL = prevFrame[1][:-1] - 2*np.dot( vectorsUnit[i-1], prevFrame[1][:-1] )*vectorsUnit[i-1]
			frL = prevFrame[0][:-1] - 2*np.dot( vectorsUnit[i-1], prevFrame[0][:-1] )*vectorsUnit[i-1]
			v2 = frame[0][:-1] - frL
			c2 = np.dot(v2,v2)
			frame[1][:-1] = upL - (2.0/c2)*np.dot(v2,upL)*v2

			frame[2][:-1] = np.cross(frame[0][:-1],frame[1][:-1])
			frame[3][:-1] = path[i]

			prevFrame = np.copy(frame)
			#sys.stderr.write("i "+str(i)+"\n")
			#sys.stderr.write("attributes[i] "+str(attributes[i])+"\n")
			frame[:-1,:-1]*=attributes[i][lightningBoltNode.LP_kRadius]
			branchFrames[i] = frame

		# multiply all the circles coordinates by the frame base coordinates along the path so we get all the points in one move yeah
		points.extend([ coord for k in range(len(circles)) for point in (np.dot(circles[k],branchFrames[k])).tolist() for coord in point  ] )

	numVertices = len(points)/4
	numFaces = len(facesCount)


	scrUtil = OpenMaya.MScriptUtil()
	scrUtil.createFromList( points, len(points))
	Mpoints = OpenMaya.MFloatPointArray( scrUtil.asDouble4Ptr(), numVertices)

	scrUtil.createFromList(facesConnect, len(facesConnect))
	MfaceConnects = OpenMaya.MIntArray( scrUtil.asIntPtr() , len(facesConnect))

	scrUtil.createFromList(facesCount, len(facesCount))
	MfacesCount = OpenMaya.MIntArray( scrUtil.asIntPtr(), numFaces)
	
	dataCreator = OpenMaya.MFnMeshData()
	outData = dataCreator.create()
	meshFS = OpenMaya.MFnMesh()


	meshFS.create(numVertices, numFaces, Mpoints , MfacesCount, MfaceConnects, outData)

	return outData

def loadGeneralLightningScript():
	# here we load also the python global script wich should be in the same folder as this plugin
	# see http://stackoverflow.com/questions/50499/in-python-how-do-i-get-the-path-and-name-of-the-file-that-is-currently-executin
	import inspect, os
	currentPluginPath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
	sys.path.append(currentPluginPath)
	if sys.modules.has_key('lightningboltProcessor') :
		del(sys.modules['lightningboltProcessor'])
		print 'module "lightningboltProcessor" deleted'
	mainLightningModule = __import__( 'lightningboltProcessor' )
	print 'module "lightningboltProcessor" loaded'

	return mainLightningModule

def setupAPVInputFromRamp( array, ramp ):
	if array.size <1 :
		raise 'array size must be more than 1'
	inc = 1.0/(array.size-1)
	param = 0.0
	valAt_util = OpenMaya.MScriptUtil()
	valAt_util.createFromDouble(1.0)
	valAtPtr = valAt_util.asFloatPtr()
	for i in range(array.size):
		ramp.getValueAtPosition(param,valAtPtr)
		array[i] = valAt_util.getFloat(valAtPtr)
		param=param+inc
	#sys.stderr.write(array)
	#sys.stderr.write('\n')

def getPointListFromCurve( numPoints, fnCurve):
	startTemp = OpenMaya.MScriptUtil()
	startTemp.createFromDouble(0.0)
	startPtr = startTemp.asDoublePtr()

	endTemp = OpenMaya.MScriptUtil()
	endTemp.createFromDouble(0.0)
	endPtr = endTemp.asDoublePtr()

	fnCurve.getKnotDomain(startPtr, endPtr)
	paramStart = startTemp.getDouble(startPtr)
	paramEnd = endTemp.getDouble(endPtr)-0.000001
	sys.stderr.write('paramEnd '+str(paramEnd)+'\n')
	
	step = (paramEnd - paramStart)/(numPoints-1)
	
	pt=OpenMaya.MPoint()
	param=paramStart
	res = []
	for i in range(numPoints):
		#sys.stderr.write('i '+str(i)+' param '+str(param)+'\n')
		fnCurve.getPointAtParam(param, pt)
		res.append( [pt.x,pt.y,pt.z] )
		param += step
	return res
		
class lightningBoltNode(OpenMayaMPx.MPxNode):
		
	#----------------------------------------------------------------------------
	##AEtemplate proc for the MRampAtributes
	mel.eval('''
global proc AElightningboltTemplate( string $nodeName )
{

	AEswatchDisplay  $nodeName;
	editorTemplate -beginScrollLayout;
		editorTemplate -beginLayout "lightning Shape parameters" -collapse 0;
			AEaddRampControl ($nodeName+".radiusRamp");
			editorTemplate -addControl "radiusMult";
			AEaddRampControl ($nodeName+".offsetRamp");
			editorTemplate -addControl "offsetMult";
		editorTemplate -endLayout;

		AEdependNodeTemplate $nodeName;

	editorTemplate -addExtraControls;
	editorTemplate -endScrollLayout;
}
			''')
	#----------------------------------------------------------------------------

	# let's have the lightningModule here as a static
	lightningModule = loadGeneralLightningScript()
	# some shortcuts
	LMProcessor = lightningModule.lightningBoltProcessor
	# shortcut to call pseudo-enum
	LP_kRadius = LMProcessor.kAPV_Radius
	LP_kIntensity = LMProcessor.kAPV_Intensity
	LP_kOffset = LMProcessor.kAPV_Offset

	# defaults
	defaultDetailValue = 2
	#----------------------------------------------------------------------------	
	# helpers to help manage attributes on a Maya node, all the helpers are "statics" methodes of the node	
	hlp = None
	#----------------------------------------------------------------------------

	def __init__(self):
		OpenMayaMPx.MPxNode.__init__(self)
		self.LightningProcessor = lightningBoltNode.LMProcessor(mayaLightningMesher)

	def compute(self, plug, data):

		if plug == lightningBoltNode.samplingDummyOut.mObject : # outDummyTrigger to Sample Ramps
			sys.stderr.write('compute dummy\n')

			thisNode = self.thisMObject()

			# Basic Ramp Attribute to Sample
			tempDetail = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.detail, thisNode,data)
			tempRadiusRamp = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.radiusRamp, thisNode,data)
			tempOffsetRamp = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.offsetRamp, thisNode,data)

			self.LightningProcessor.setDetail(tempDetail)
			setupAPVInputFromRamp( self.LightningProcessor.getAPVariation1( lightningBoltNode.LP_kRadius ), tempRadiusRamp )
			setupAPVInputFromRamp( self.LightningProcessor.getAPVariation1( lightningBoltNode.LP_kOffset ), tempOffsetRamp )

			outputHandle = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.samplingDummyOut, thisNode,data)
			outputHandle.setInt(1)
			data.setClean(plug)

		elif plug == lightningBoltNode.outputMesh.mObject : # outMesh
			sys.stderr.write('compute mesh\n')

			thisNode = self.thisMObject()

			tempTimeShape = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.timeShape, thisNode,data)
			tempTimeBranching = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.timeBranching, thisNode,data)
			tempSeedShape = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.seedShape, thisNode,data)
			tempSeedBranching = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.seedBranching, thisNode,data)
			tempRadiusMult = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.radiusMult, thisNode,data)
			tempOffsetMult = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.offsetMult, thisNode,data)


			# force evaluation if needed of ramp samples (this will trigger the plug for outDummyAtt of the compute)
			plugDum = OpenMaya.MPlug( thisNode, lightningBoltNode.samplingDummyOut.mObject) 
			triggerSampling = plugDum.asInt()

			'''
			# Chaos Offset Attribute
			trunkChaosOffsetRamp = lightningBoltNode.getAttValue(thisNode,data,"trunkChaosOffsetRamp")
			tempChaosOffsetMax = lightningBoltNode.getAttValue(thisNode,data,"chaosOffsetMax")
			# Proba Ramp Attribute
			trunkProbaRamp = lightningBoltNode.getAttValue(thisNode,data,"trunkProbaRamp")
			tempProbaMax = lightningBoltNode.getAttValue(thisNode,data,"probaMax")
			# Elevation Angle Ramp Attribute
			trunkElevationAngleRamp = lightningBoltNode.getAttValue(thisNode,data,"trunkElevationAngleRamp")
			# Max Angle Ramp Attribute
			trunkAngleRamp = lightningBoltNode.getAttValue(thisNode,data,"trunkMaxAngleRamp")
			tempMaxAngleMax = lightningBoltNode.getAttValue(thisNode,data,"maxAngleMax")
			#			------------------------
			
			tempIteration = lightningBoltNode.getAttValue(thisNode,data,"iterations")
			tempTubeDetail = lightningBoltNode.getAttValue(thisNode,data,"tubeSides")
			tempVectorStart = lightningBoltNode.getAttValue(thisNode,data,"upVectorStart")
			'''			
			# load the inputs of the node into the processor			
			self.LightningProcessor.setAPVMult1( lightningBoltNode.LP_kRadius, tempRadiusMult )
			self.LightningProcessor.setAPVMult1( lightningBoltNode.LP_kOffset, tempOffsetMult )

			#sys.stderr.write("vector "+str(tempVectorStart.x)+" "+str(tempVectorStart.y)+" "+str(tempVectorStart.z)+"\n")

			#fnDep = OpenMaya.MFnDependencyNode(thisNode)
			#nodeName = fnDep.name()
			#sys.stderr.write('node '+nodeName+' has object '+str(self.LightningProcessor)+'\n')
			
			tempCurve = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.inputCurve, thisNode,data)

			if tempCurve.isNull():
				sys.stderr.write('there is no curve!\n')
				data.setClean(plug)
				return

			fnNurbs = OpenMaya.MFnNurbsCurve( tempCurve )
			#numPoints = fnNurbs.numCVs()
			#points = OpenMaya.MPointArray()
			#fnNurbs.getCVs(points)
					
			outputHandle = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.outputMesh, thisNode,data)

			pointList = getPointListFromCurve( self.LightningProcessor.maxPoints, fnNurbs )
			#pointList = [ [p.x,p.y,p.z] for p in points ]
			self.LightningProcessor.initializeProcessor()
			self.LightningProcessor.addSeedPointList(pointList)
			outputData = self.LightningProcessor.process()

			sys.stderr.write('end of compute\n')

			outputHandle.setMObject(outputData)
			data.setClean(plug)
		else:
			return OpenMaya.kUnknownParameter

	def postConstructorRampValueInitialize( self, rampMObj):
		thisNode = self.thisMObject()
		plugRamp = OpenMaya.MPlug( thisNode, rampMObj)
		elemPlug = plugRamp.elementByLogicalIndex( 0 )
		posPlug = elemPlug.child(0)
		valPlug = elemPlug.child(1)
		valPlug.setDouble(1)
		posPlug.setDouble(0)

	def postConstructor(self):
		# setting the Ramp initial value because it cannot be done in the AETemplate
		# thanks to this blog for the idea : http://www.chadvernon.com/blog/resources/maya-api-programming/mrampattribute/
		self.postConstructorRampValueInitialize(lightningBoltNode.radiusRamp.mObject)
		self.postConstructorRampValueInitialize(lightningBoltNode.offsetRamp.mObject)

def nodeCreator():
	return OpenMayaMPx.asMPxPtr( lightningBoltNode() )

def nodeInitializer():
	unitAttr = OpenMaya.MFnUnitAttribute()
	typedAttr = OpenMaya.MFnTypedAttribute()
	numAttr = OpenMaya.MFnNumericAttribute()
	compAttr = OpenMaya.MFnCompoundAttribute()

	lightningBoltNode.hlp = attCreationHlp(lightningBoltNode)

	lightningBoltNode.hlp.createAtt( name = "timeShape", fn=unitAttr, shortName="ts", type=OpenMaya.MFnUnitAttribute.kTime, default=0.0, exceptAffectList=['samplingDummyOut'] )
	lightningBoltNode.hlp.createAtt( name = "timeBranching", fn=unitAttr, shortName="tb", type=OpenMaya.MFnUnitAttribute.kTime, default=0.0, exceptAffectList=['samplingDummyOut'] )	
	lightningBoltNode.hlp.createAtt( name = "seedShape", fn=numAttr, shortName="ss", type=OpenMaya.MFnNumericData.kInt, default=0.0, exceptAffectList=['samplingDummyOut'] )	
	lightningBoltNode.hlp.createAtt( name = "seedBranching", fn=numAttr, shortName="sb", type=OpenMaya.MFnNumericData.kInt, default=0.0, exceptAffectList=['samplingDummyOut'] )	
	lightningBoltNode.hlp.createAtt( name = "inputCurve", fn=typedAttr, shortName="ic", type=OpenMaya.MFnData.kNurbsCurve, exceptAffectList=['samplingDummyOut'] )	

	lightningBoltNode.hlp.createAtt( name = "detail", fn=numAttr, shortName="det", type=OpenMaya.MFnNumericData.kInt, default=lightningBoltNode.defaultDetailValue )	
	lightningBoltNode.hlp.createAtt( name = "radiusRamp", fn=OpenMaya.MRampAttribute, shortName="rr", type="Ramp" )		
	lightningBoltNode.hlp.createAtt( name = "radiusMult", fn=numAttr, shortName="rm", type=OpenMaya.MFnNumericData.kDouble, default=0.5, exceptAffectList=['samplingDummyOut'] )	

	lightningBoltNode.hlp.createAtt( name = "offsetRamp", fn=OpenMaya.MRampAttribute, shortName="or", type="Ramp" )		
	lightningBoltNode.hlp.createAtt( name = "offsetMult", fn=numAttr, shortName="om", type=OpenMaya.MFnNumericData.kDouble, default=1.0, exceptAffectList=['samplingDummyOut'] )	

#	lightningBoltNode.createAtt( name = "timeShape", fn=unitAttr, shortName="ts", type=OpenMaya.MFnUnitAttribute.kTime, default=0.0, outsAffectList=['outputMesh'] )	
#	lightningBoltNode.createAtt( name = "timeBranching", fn=unitAttr, shortName="tb", type=OpenMaya.MFnUnitAttribute.kTime, default=0.0, outsAffectList=['outputMesh'] )	

#	lightningBoltNode.createAtt( name = "seedShape", fn=numAttr, shortName="ss", type=OpenMaya.MFnNumericData.kInt, default=0.0, outsAffectList=['outputMesh'] )	
#	lightningBoltNode.createAtt( name = "seedBranching", fn=numAttr, shortName="sb", type=OpenMaya.MFnNumericData.kInt, default=0.0, outsAffectList=['outputMesh'] )	
#	lightningBoltNode.createAtt( name = "inputCurve", fn=typedAttr, shortName="ic", type=OpenMaya.MFnData.kNurbsCurve, outsAffectList=['outputMesh'] )	

#	lightningBoltNode.createAtt( name = "detail", fn=numAttr, shortName="det", type=OpenMaya.MFnNumericData.kInt, default=lightningBoltNode.defaultDetailValue )	
#	lightningBoltNode.createAtt( name = "radiusRamp", fn=OpenMaya.MRampAttribute, shortName="rr", type="Ramp" )		
#	lightningBoltNode.createAtt( name = "radiusMult", fn=numAttr, shortName="rm", type=OpenMaya.MFnNumericData.kDouble, default=0.5, outsAffectList=['outputMesh'] )	
	
	

	'''
	lightningBoltNode.createAtt( name = "trunkChaosOffsetRamp", fn=OpenMaya.MRampAttribute, shortName="tcor", type="Ramp" )	
	lightningBoltNode.createAtt( name = "chaosOffsetMax", fn=numAttr, shortName="co", type=OpenMaya.MFnNumericData.kDouble, default=0.6 )	
	
	lightningBoltNode.createAtt( name = "trunkProbaRamp", fn=OpenMaya.MRampAttribute, shortName="tpr", type="Ramp" )	
	lightningBoltNode.createAtt( name = "probaMax", fn=numAttr, shortName="pm", type=OpenMaya.MFnNumericData.kDouble, default=0.5 )	

	lightningBoltNode.createAtt( name = "trunkElevationAngleRamp", fn=OpenMaya.MRampAttribute, shortName="tear", type="Ramp" )	
	lightningBoltNode.createAtt( name = "trunkMaxAngleRamp", fn=OpenMaya.MRampAttribute, shortName="tmax", type="Ramp" )	
	lightningBoltNode.createAtt( name = "maxAngleMax", fn=numAttr, shortName="mam", type=OpenMaya.MFnNumericData.kDouble, default=0.5 )	
	
	lightningBoltNode.createAtt( name = "iterations", fn=numAttr, shortName="it", type=OpenMaya.MFnNumericData.kInt, default=2 )	
	lightningBoltNode.createAtt( name = "tubeSides", fn=numAttr, shortName="td", type=OpenMaya.MFnNumericData.kInt, default=1 )	

	lightningBoltNode.createAtt( name = "upVectorStartX", fn=numAttr, shortName="upsX", type=OpenMaya.MFnNumericData.kDouble, default=0.0, addAttr=False )
	lightningBoltNode.createAtt( name = "upVectorStartY", fn=numAttr, shortName="upsY", type=OpenMaya.MFnNumericData.kDouble, default=1.0, addAttr=False )
	lightningBoltNode.createAtt( name = "upVectorStartZ", fn=numAttr, shortName="upsZ", type=OpenMaya.MFnNumericData.kDouble, default=0.0, addAttr=False )
	lightningBoltNode.createAtt( name = "upVectorStart", fn=numAttr, shortName="ups", type="vectorOfDouble", childs=["upVectorStartX","upVectorStartY","upVectorStartZ"] )
	'''
	lightningBoltNode.hlp.createAtt( name = "samplingDummyOut", isInput=False, fn=numAttr, shortName="sdo", type=OpenMaya.MFnNumericData.kInt)
	lightningBoltNode.hlp.createAtt( name = "outputMesh", isInput=False, fn=typedAttr, shortName="out", type=OpenMaya.MFnData.kMesh)	
	lightningBoltNode.hlp.addAllAttrs()
	lightningBoltNode.hlp.generateAffects()



#	lightningBoltNode.createAtt( name = "samplingDummyOut", isInput=False, fn=numAttr, shortName="sdo", type=OpenMaya.MFnNumericData.kInt)

#	lightningBoltNode.createAtt( name = "outputMesh", isInput=False, fn=typedAttr, shortName="out", type=OpenMaya.MFnData.kMesh)	

	#lightningBoltNode.addAllAttrs()
	#lightningBoltNode.generateAffects()
def cleanupClass():
	delattr(lightningBoltNode,'defaultDetailValue')
	delattr(lightningBoltNode,'lightningModule')
	delattr(lightningBoltNode,'LMProcessor')
	delattr(lightningBoltNode,'LP_kRadius')
	delattr(lightningBoltNode,'LP_kIntensity')
	delattr(lightningBoltNode,'LP_kOffset')
	delattr(lightningBoltNode,'hlp')

# initialize the script plug-in
def initializePlugin(mobject):
	mplugin = OpenMayaMPx.MFnPlugin(mobject)
	try:
		mplugin.registerNode( kPluginNodeName, kPluginNodeId, nodeCreator, nodeInitializer)
	except:
		sys.stderr.write( "Failed to register node: %s" % kPluginNodeName )
		raise

# uninitialize the script plug-in
def uninitializePlugin(mobject):
	cleanupClass()
	mplugin = OpenMayaMPx.MFnPlugin(mobject)
	try:
		mplugin.deregisterNode( kPluginNodeId )
	except:
		sys.stderr.write( "Failed to deregister node: %s" % kPluginNodeName )
		raise
