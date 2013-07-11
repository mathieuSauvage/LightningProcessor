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

# TODO Add a check of the short names for attributes
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

def mayaLightningMesher(resultLoopingPoints, numLoopingLightningBranch, resultPoints, numLightningBranch, resultIntensityColors, branchMaxPoints, tubeSide):
	
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
	meshFn = OpenMaya.MFnMesh()


	meshFn.create(numVertices, numFaces, Mpoints , MfacesCount, MfaceConnects, outData)

	# vertexColors
	scrUtil.createFromList( resultIntensityColors, len(resultIntensityColors))
	MColors = OpenMaya.MColorArray( scrUtil.asDouble4Ptr(), numVertices )

	scrUtil.createFromList( range(numVertices) , numVertices )
	MVertexIds = OpenMaya.MIntArray( scrUtil.asIntPtr(), numVertices )

	meshFn.setVertexColors( MColors, MVertexIds )
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
	#sys.stderr.write('paramEnd '+str(paramEnd)+'\n')
	
	step = (paramEnd - paramStart)/(numPoints-1)
	
	pt=OpenMaya.MPoint()
	param=paramStart
	res = []
	fullLength = 0
	for i in range(numPoints):
		#sys.stderr.write('i '+str(i)+' param '+str(param)+'\n')
		fnCurve.getPointAtParam(param, pt)
		res.append( [pt.x,pt.y,pt.z] )
		if i>0 :
			fullLength += math.sqrt(res[i-1][0]*pt.x + res[i-1][1]*pt.y + res[i-1][2]*pt.z) 
		#sys.stderr.write('fullLength '+str(fullLength)+'\n')

		param += step
	return res, fnCurve.length()
		
class lightningBoltNode(OpenMayaMPx.MPxNode):
		
	#----------------------------------------------------------------------------
	##AEtemplate proc for the MRampAtributes
	mel.eval('''
global proc AElightningboltTemplate( string $nodeName )
{

	AEswatchDisplay  $nodeName;
	editorTemplate -beginScrollLayout;

		editorTemplate -beginLayout "global parameters" -collapse 0;
			editorTemplate -addControl "tubeSides";
			editorTemplate -addControl "maxGeneration";
			editorTemplate -addControl "detail";
			editorTemplate -addControl "seedShape";
			editorTemplate -addControl "seedBranching";
		editorTemplate -endLayout;

		editorTemplate -beginLayout "lightning Shape parameters" -collapse 0;
			editorTemplate -addControl "timeShape";
			editorTemplate -addControl "timeShapeMultiplier";
			editorTemplate -addControl "timeBranching";
			editorTemplate -addControl "timeBranchingMultiplier";
			editorTemplate -addControl "shapeFrequency";
			editorTemplate -addControl "numChildren";
			editorTemplate -addControl "numChildrenRand";			
			
			AEaddRampControl ($nodeName+".radiusRamp");
			editorTemplate -addControl "radiusMult";
			AEaddRampControl ($nodeName+".offsetRamp");
			editorTemplate -addControl "offsetMult";
			AEaddRampControl ($nodeName+".childLengthRamp");
			editorTemplate -addControl "childLengthMult";
			AEaddRampControl ($nodeName+".intensityRamp");
			editorTemplate -addControl "intensityMult";
			AEaddRampControl ($nodeName+".elevationRamp");
			editorTemplate -addControl "elevationMult";
			AEaddRampControl ($nodeName+".elevationRandRamp");
			editorTemplate -addControl "elevationRandMult";
		editorTemplate -endLayout;

		editorTemplate -beginLayout "generation transfert multipliers" -collapse 0;
			editorTemplate -addControl "transfertTimeBranching";
			editorTemplate -addControl "transfertTimeShape";
			editorTemplate -addControl "transfertRadius";
			editorTemplate -addControl "transfertOffset";			

		editorTemplate -endLayout;

		AEdependNodeTemplate $nodeName;

	editorTemplate -addExtraControls;
	editorTemplate -endScrollLayout;
}
			''')
	#----------------------------------------------------------------------------

	# let's have the lightningModule here as a static
	lightningModule = loadGeneralLightningScript()
	# shortcuts
	LMLP = lightningModule.lightningBoltProcessor

	# defaults
	defaultDetailValue = 1
	defaultMaxGeneration = 0
	defaultNumChildren = 2
	defaultNumChildrenRand = 2
	defaultTubeSides = 4

	#----------------------------------------------------------------------------	
	# helpers to help manage attributes on a Maya node, all the helpers are "statics" methodes of the node	
	hlp = None
	#----------------------------------------------------------------------------

	def __init__(self):
		OpenMayaMPx.MPxNode.__init__(self)
		self.LightningProcessor = lightningBoltNode.LMLP(mayaLightningMesher)

	def compute(self, plug, data):

		if plug == lightningBoltNode.samplingDummyOut.mObject : # outDummyTrigger to Sample Ramps
			sys.stderr.write('compute dummy\n')

			thisNode = self.thisMObject()

			# Basic Ramp Attribute to Sample
			tempDetail = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.detail, thisNode,data)
			tempRadiusRamp = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.radiusRamp, thisNode,data)
			tempIntensityRamp = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.intensityRamp, thisNode,data)
			tempOffsetRamp = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.offsetRamp, thisNode,data)
			tempChildLengthRamp = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.childLengthRamp, thisNode,data)
			tempElevationRamp = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.elevationRamp, thisNode,data)
			tempElevationRandRamp = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.elevationRandRamp, thisNode,data)

			self.LightningProcessor.setDetail(tempDetail)
			setupAPVInputFromRamp( self.LightningProcessor.getAPVariation1( lightningBoltNode.LMLP.eAPBR.radius ), tempRadiusRamp )
			setupAPVInputFromRamp( self.LightningProcessor.getAPVariation1( lightningBoltNode.LMLP.eAPBR.intensity ), tempIntensityRamp )
			#setupAPVInputFromRamp( self.LightningProcessor.getAPVariation1( lightningBoltNode.LMLP.eAPBR.offset ), tempOffsetRamp )
			setupAPVInputFromRamp( self.LightningProcessor.getAPVariation1( lightningBoltNode.LMLP.eAPBR.childLength ), tempChildLengthRamp )
			#setupAPVInputFromRamp( self.LightningProcessor.getAPVariation1( lightningBoltNode.LMLP.eAPBR.elevation ), tempElevationRamp )
			#setupAPVInputFromRamp( self.LightningProcessor.getAPVariation1( lightningBoltNode.LMLP.eAPBR.elevationRand ), tempElevationRandRamp )

			setupAPVInputFromRamp( self.LightningProcessor.getAPSPE1( lightningBoltNode.LMLP.eAPSPE.offset ), tempOffsetRamp )

			setupAPVInputFromRamp( self.LightningProcessor.getAPSPE1( lightningBoltNode.LMLP.eAPSPE.elevation ), tempElevationRamp )
			setupAPVInputFromRamp( self.LightningProcessor.getAPSPE1( lightningBoltNode.LMLP.eAPSPE.elevationRand ), tempElevationRandRamp )


			outputHandle = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.samplingDummyOut, thisNode,data)
			outputHandle.setInt(1)
			data.setClean(plug)

		elif plug == lightningBoltNode.outputMesh.mObject : # outMesh
			sys.stderr.write('compute mesh\n')

			thisNode = self.thisMObject()

			# generic values
			tempTubeSides = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.tubeSides, thisNode,data)

			tempMaxGeneration = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.maxGeneration, thisNode,data)
			tempShapeFrequency = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.shapeFrequency, thisNode,data)
			tempNumChildren = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.numChildren, thisNode, data)
			tempNumChildrenRand = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.numChildrenRand, thisNode, data)

			tempTimeShape = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.timeShape, thisNode,data)
			tempTimeShapeMultiplier = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.timeShapeMultiplier, thisNode,data)

			tempTimeBranching = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.timeBranching, thisNode,data)
			tempTimeBranchingMultiplier= lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.timeBranchingMultiplier, thisNode,data)

			tempSeedShape = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.seedShape, thisNode,data)
			tempSeedBranching = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.seedBranching, thisNode,data)
			tempRadiusMult = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.radiusMult, thisNode,data)
			tempIntensityMult = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.intensityMult, thisNode,data)
			tempOffsetMult = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.offsetMult, thisNode,data)
			tempChildLengthMult = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.childLengthMult, thisNode,data)
			tempElevationMult = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.elevationMult, thisNode,data)
			tempElevationRandMult = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.elevationRandMult, thisNode,data)

			# transfert values
			tempTransfertRadius = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.transfertRadius, thisNode,data)
			tempTransfertOffset = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.transfertOffset, thisNode,data)
			tempTransfertTimeBranching = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.transfertTimeBranching, thisNode,data)
			tempTransfertTimeShape = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.transfertTimeShape, thisNode,data)


			# force evaluation if needed of ramp samples (this will trigger the plug for outDummyAtt of the compute)
			plugDum = OpenMaya.MPlug( thisNode, lightningBoltNode.samplingDummyOut.mObject) 
			triggerSampling = plugDum.asInt()


			outputHandle = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.outputMesh, thisNode,data)

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
			tempCurve = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.inputCurve, thisNode,data)

			if tempCurve.isNull():
				sys.stderr.write('there is no curve!\n')
				data.setClean(plug)
				return

			fnNurbs = OpenMaya.MFnNurbsCurve( tempCurve )
			pointList, cvLength = getPointListFromCurve( self.LightningProcessor.maxPoints, fnNurbs )

			sys.stderr.write('child length '+str(tempChildLengthMult*cvLength)+'\n')
			# load the Along Path Values inputs of the node into the processor			
			self.LightningProcessor.setAPVMult1( lightningBoltNode.LMLP.eAPBR.radius, tempRadiusMult )
			self.LightningProcessor.setAPVMult1( lightningBoltNode.LMLP.eAPBR.intensity, tempIntensityMult )
			#self.LightningProcessor.setAPVMult1( lightningBoltNode.LMLP.eAPBR.offset, tempOffsetMult )
			#self.LightningProcessor.setAPVMult1( lightningBoltNode.LMLP.eAPBR.elevation, tempElevationMult )
			#self.LightningProcessor.setAPVMult1( lightningBoltNode.LMLP.eAPBR.elevationRand, tempElevationRandMult )
			self.LightningProcessor.setAPVMult1( lightningBoltNode.LMLP.eAPBR.childLength, tempChildLengthMult*cvLength )

			# load the Values inputs
			self.LightningProcessor.setValue( lightningBoltNode.LMLP.eGEN.shapeFrequency, tempShapeFrequency )

			self.LightningProcessor.setValue( lightningBoltNode.LMLP.eGEN.shapeTimeMult, tempTimeShapeMultiplier )
			self.LightningProcessor.setValue( lightningBoltNode.LMLP.eGEN.branchingTimeMult, tempTimeBranchingMultiplier )
			self.LightningProcessor.setValue( lightningBoltNode.LMLP.eGEN.numChildrens, tempNumChildren )
			self.LightningProcessor.setValue( lightningBoltNode.LMLP.eGEN.randNumChildrens, tempNumChildrenRand )
			self.LightningProcessor.setValue( lightningBoltNode.LMLP.eGEN.offset, tempOffsetMult )

			# load the Generic values
			self.LightningProcessor.setSpecialBranchValue( lightningBoltNode.LMLP.eSPEBR.seedShape, tempSeedShape )
			self.LightningProcessor.setSpecialBranchValue( lightningBoltNode.LMLP.eSPEBR.seedBranching, tempSeedBranching )

			# load the Transfert values
			self.LightningProcessor.setValueTransfert( lightningBoltNode.LMLP.eGEN.branchingTimeMult, tempTransfertTimeBranching )
			self.LightningProcessor.setValueTransfert( lightningBoltNode.LMLP.eGEN.shapeTimeMult, tempTransfertTimeShape )
			self.LightningProcessor.setValueTransfert( lightningBoltNode.LMLP.eGEN.offset, tempTransfertOffset )

			self.LightningProcessor.setAPVTransfert( lightningBoltNode.LMLP.eAPBR.radius, tempTransfertRadius )
			#self.LightningProcessor.setAPVTransfert( lightningBoltNode.LMLP.eAPBR.offset, tempTransfertOffset )



			#sys.stderr.write("vector "+str(tempVectorStart.x)+" "+str(tempVectorStart.y)+" "+str(tempVectorStart.z)+"\n")

			#fnDep = OpenMaya.MFnDependencyNode(thisNode)
			#nodeName = fnDep.name()
			#sys.stderr.write('node '+nodeName+' has object '+str(self.LightningProcessor)+'\n')
			

			self.LightningProcessor.initializeProcessor()
			self.LightningProcessor.addSeedPointList(pointList)
			outputData = self.LightningProcessor.process(tempMaxGeneration, tempTimeBranching.value(), tempTimeShape.value(), tempTubeSides)

			sys.stderr.write('end of compute\n')

			outputHandle.setMObject(outputData)
			data.setClean(plug)
		else:
			return OpenMaya.kUnknownParameter

	def postConstructorRampInitialize( self, rampMObj, values):
		thisNode = self.thisMObject()
		plugRamp = OpenMaya.MPlug( thisNode, rampMObj)
		for i in range(len(values)) :
			val = values[i]
			elemPlug = plugRamp.elementByLogicalIndex( i )
			posPlug = elemPlug.child(0)
			valPlug = elemPlug.child(1)
			posPlug.setDouble(val[0])
			valPlug.setDouble(val[1])
	
	def postConstructor(self):
		# setting the Ramp initial value because it cannot be done in the AETemplate
		# thanks to this blog for the idea : http://www.chadvernon.com/blog/resources/maya-api-programming/mrampattribute/
		self.postConstructorRampInitialize( lightningBoltNode.radiusRamp.mObject, [(0,1),(1,0)] )
		self.postConstructorRampInitialize( lightningBoltNode.offsetRamp.mObject, [(0,0),(.05,1)] )
		self.postConstructorRampInitialize( lightningBoltNode.childLengthRamp.mObject, [(0,1),(1,.25)] )
		self.postConstructorRampInitialize( lightningBoltNode.intensityRamp.mObject, [(0,1),(1,0.5)] )
		self.postConstructorRampInitialize( lightningBoltNode.elevationRamp.mObject, [(0,0.5),(1,.15)] )
		self.postConstructorRampInitialize( lightningBoltNode.elevationRandRamp.mObject, [(0,1)] )

def nodeCreator():
	return OpenMayaMPx.asMPxPtr( lightningBoltNode() )

def nodeInitializer():
	unitAttr = OpenMaya.MFnUnitAttribute()
	typedAttr = OpenMaya.MFnTypedAttribute()
	numAttr = OpenMaya.MFnNumericAttribute()
	compAttr = OpenMaya.MFnCompoundAttribute()

	lightningBoltNode.hlp = attCreationHlp(lightningBoltNode)

	lightningBoltNode.hlp.createAtt( name = "inputCurve", fn=typedAttr, shortName="ic", type=OpenMaya.MFnData.kNurbsCurve, exceptAffectList=['samplingDummyOut'] )	

	lightningBoltNode.hlp.createAtt( name = "tubeSides", fn=numAttr, shortName="tus", type=OpenMaya.MFnNumericData.kInt, default=lightningBoltNode.defaultTubeSides, exceptAffectList=['samplingDummyOut'] )	

	lightningBoltNode.hlp.createAtt( name = "maxGeneration", fn=numAttr, shortName="mg", type=OpenMaya.MFnNumericData.kInt, default=lightningBoltNode.defaultMaxGeneration, exceptAffectList=['samplingDummyOut'] )	
	lightningBoltNode.hlp.createAtt( name = "detail", fn=numAttr, shortName="det", type=OpenMaya.MFnNumericData.kInt, default=lightningBoltNode.defaultDetailValue )
	lightningBoltNode.hlp.createAtt( name = "shapeFrequency", fn=numAttr, shortName="sf", type=OpenMaya.MFnNumericData.kDouble, default=1.0, exceptAffectList=['samplingDummyOut'] )	

	lightningBoltNode.hlp.createAtt( name = "numChildren", fn=numAttr, shortName="nc", type=OpenMaya.MFnNumericData.kInt, default=lightningBoltNode.defaultNumChildren, exceptAffectList=['samplingDummyOut'] )	
	lightningBoltNode.hlp.createAtt( name = "numChildrenRand", fn=numAttr, shortName="ncr", type=OpenMaya.MFnNumericData.kInt, default=lightningBoltNode.defaultNumChildrenRand, exceptAffectList=['samplingDummyOut'] )	

	lightningBoltNode.hlp.createAtt( name = "timeShape", fn=unitAttr, shortName="ts", type=OpenMaya.MFnUnitAttribute.kTime, default=0.0, exceptAffectList=['samplingDummyOut'] )
	lightningBoltNode.hlp.createAtt( name = "timeShapeMultiplier", fn=numAttr, shortName="tsm", type=OpenMaya.MFnNumericData.kDouble, default=1.0, exceptAffectList=['samplingDummyOut'] )	


	lightningBoltNode.hlp.createAtt( name = "timeBranching", fn=unitAttr, shortName="tb", type=OpenMaya.MFnUnitAttribute.kTime, default=0.0, exceptAffectList=['samplingDummyOut'] )
	lightningBoltNode.hlp.createAtt( name = "timeBranchingMultiplier", fn=numAttr, shortName="tbm", type=OpenMaya.MFnNumericData.kDouble, default=1.0, exceptAffectList=['samplingDummyOut'] )	


	lightningBoltNode.hlp.createAtt( name = "seedShape", fn=numAttr, shortName="ss", type=OpenMaya.MFnNumericData.kInt, default=0.0, exceptAffectList=['samplingDummyOut'] )	
	lightningBoltNode.hlp.createAtt( name = "seedBranching", fn=numAttr, shortName="sb", type=OpenMaya.MFnNumericData.kInt, default=0.0, exceptAffectList=['samplingDummyOut'] )	

	lightningBoltNode.hlp.createAtt( name = "radiusRamp", fn=OpenMaya.MRampAttribute, shortName="rr", type="Ramp" )		
	lightningBoltNode.hlp.createAtt( name = "radiusMult", fn=numAttr, shortName="rm", type=OpenMaya.MFnNumericData.kDouble, default=0.25, exceptAffectList=['samplingDummyOut'] )	

	lightningBoltNode.hlp.createAtt( name = "offsetRamp", fn=OpenMaya.MRampAttribute, shortName="or", type="Ramp" )		
	lightningBoltNode.hlp.createAtt( name = "offsetMult", fn=numAttr, shortName="om", type=OpenMaya.MFnNumericData.kDouble, default=1.0, exceptAffectList=['samplingDummyOut'] )	

	lightningBoltNode.hlp.createAtt( name = "childLengthRamp", fn=OpenMaya.MRampAttribute, shortName="clr", type="Ramp" )
	lightningBoltNode.hlp.createAtt( name = "childLengthMult", fn=numAttr, shortName="clm", type=OpenMaya.MFnNumericData.kDouble, default=1.0, exceptAffectList=['samplingDummyOut'] )	
	
	lightningBoltNode.hlp.createAtt( name = "intensityRamp", fn=OpenMaya.MRampAttribute, shortName="ir", type="Ramp" )
	lightningBoltNode.hlp.createAtt( name = "intensityMult", fn=numAttr, shortName="im", type=OpenMaya.MFnNumericData.kDouble, default=1.0, exceptAffectList=['samplingDummyOut'] )	

	# elevation 1 = 180 degree
	lightningBoltNode.hlp.createAtt( name = "elevationRamp", fn=OpenMaya.MRampAttribute, shortName="er", type="Ramp" )
	lightningBoltNode.hlp.createAtt( name = "elevationMult", fn=numAttr, shortName="em", type=OpenMaya.MFnNumericData.kDouble, default=.33, exceptAffectList=['samplingDummyOut'] )	

	lightningBoltNode.hlp.createAtt( name = "elevationRandRamp", fn=OpenMaya.MRampAttribute, shortName="err", type="Ramp" )
	lightningBoltNode.hlp.createAtt( name = "elevationRandMult", fn=numAttr, shortName="erm", type=OpenMaya.MFnNumericData.kDouble, default=.15, exceptAffectList=['samplingDummyOut'] )	


	# Transfert multipliers
	lightningBoltNode.hlp.createAtt( name = "transfertTimeBranching", fn=numAttr, shortName="ttb", type=OpenMaya.MFnNumericData.kDouble, default=3.0, exceptAffectList=['samplingDummyOut'] )
	lightningBoltNode.hlp.createAtt( name = "transfertTimeShape", fn=numAttr, shortName="tts", type=OpenMaya.MFnNumericData.kDouble, default=1.0, exceptAffectList=['samplingDummyOut'] )	

	lightningBoltNode.hlp.createAtt( name = "transfertRadius", fn=numAttr, shortName="tr", type=OpenMaya.MFnNumericData.kDouble, default=1.0, exceptAffectList=['samplingDummyOut'] )	
	lightningBoltNode.hlp.createAtt( name = "transfertOffset", fn=numAttr, shortName="to", type=OpenMaya.MFnNumericData.kDouble, default=1.0, exceptAffectList=['samplingDummyOut'] )	


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
	delattr(lightningBoltNode,'LMLP')
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
