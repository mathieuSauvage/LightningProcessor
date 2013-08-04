#!/usr/bin/python
# -*- coding: iso-8859-1 -*-

'''
================================================================================
* VERSION 1.0
================================================================================
* AUTHOR:
Mathieu Sauvage mathieu@hiddenforest.fr
================================================================================
* INTERNET SOURCE:
================================================================================
* DESCRIPTION:
This is the Maya plugin for the lightning Processor
================================================================================
* DEPENDENCIES:
- You need numpy installed in your Maya's Python in order to execute the lightning
Processor
- You need pymel installed just in order to execute the creation script below
================================================================================
* USAGE:
- if you want to put the plugin into the Maya Folder, be sure to move this file and lightningboltProcessor.py together
- the AEtemplate is embedded in this file, so you don't need to install it
- load the plugin in Maya, if you didn't move the files in a Maya Folder you can just execute
pm.loadPlugin('{YOUR LIGHTNING PLUGIN PATH}/mayaLightningPlugin.py')
- select some curves and execute this creation script in the python script Editor:

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

curvesToLightning( pm.ls(sl=True) )

================================================================================
* TODO:
================================================================================
'''

import sys
import maya.OpenMaya as OpenMaya
import maya.OpenMayaMPx as OpenMayaMPx
import maya.mel as mel

#-------------------------------------------------------------------------------
# Here is the Helper Class to make to process of creating, accessing and setting
# the influence of attributes less tedious

def enum(*sequential, **named):
	enums = dict(zip(sequential, range(len(sequential))), **named)
	return type('Enum', (), enums)

class MAH_msCommandException(Exception):
    def __init__(self,message):
        self.message = '[MAH] '+message
    
    def __str__(self):
        return self.message

class attDef:
	def __init__( self, index, mObject, name , shortName, type,  addAttr, isInput, isArray  ):
		self.mObject = mObject
		self.index = index
		self.name = name
		self.shortName = shortName
		self.type = type
		self.addAttr = addAttr
		self.isInput = isInput
		self.isArray = isArray

class computeGroup:
	def __init__( self, id ):
		self.OUTS = []
		self.INS = []
		self.id = id
		self.dependencies = []
		self.affect = []

	def addIN( self, index ):
		self.INS.append( index )

	def addOUT( self, index ):
		self.OUTS.append( index )

	def addAffect( self, computeGp ):
		if computeGp in self.affect:
			raise MAH_msCommandException( 'computeGroup: group [ '+self.id+' ] already affect [ '+computeGp.id+' ], error')
		self.affect.append(computeGp)

	def addDependencies( self, dependencies ):
		for d in dependencies:
			if d in self.dependencies:
				raise MAH_msCommandException('computeGroup: group [ '+self.id+' ] depend already on [ '+d.id+' ], error')
			self.dependencies.append( d )
			d.addAffect( self )

	def forceEvaluateOUTS( self, thisNode ):
		for attOut in self.OUTS :
			plugDummy = OpenMaya.MPlug( thisNode, attOut.mObject ) 
			trigger = plugDummy.asInt()

	def checkForPlug( self, thisNode, plug ):
		needCompute = False
		for attOut in self.OUTS:
			if plug == attOut.mObject:
				needCompute = True
				break
		if needCompute:
			for dep in self.dependencies :
				dep.forceEvaluateOUTS(thisNode)
		return needCompute

	def setClean( self, data ):
		for attOut in self.OUTS:
			data.setClean( attOut.mObject )

eHlpT = enum('double', 'int', 'bool', 'time', 'curve', 'mesh', 'ramp', 'maxType')
aHlpMayaTypes = [ OpenMaya.MFnNumericData.kDouble, OpenMaya.MFnNumericData.kInt, OpenMaya.MFnNumericData.kBoolean, OpenMaya.MFnUnitAttribute.kTime, OpenMaya.MFnData.kNurbsCurve, OpenMaya.MFnData.kMesh, '' ]
eHlpDiscBehavior = enum( 'nothing', 'delete', 'reset')
aHlpMayaDiscBehavior = [ OpenMaya.MFnAttribute.kNothing, OpenMaya.MFnAttribute.kDelete, OpenMaya.MFnAttribute.kReset ]

class MayaAttHelper:	
	def __init__(self,nodeClass, numComputeGroups):
		self.attributes = []
		self.nodeClass = nodeClass
		self.computeGroups = [None]*numComputeGroups
		self.dummyNames = {} # we keep track of dummy's names to not have duplicates during creation of the dummy attribute

	def cleanUp( self ):
		for att in self.attributes:
			delattr(self.nodeClass, att.name+'Index' )
		self.computeGroups = None
		self.attributes = None
		self.nodeClass = None

	def createAtt( self, name, shortName, type, computeGpIds, isInput=True, default=None, addAttr=True, isArray=False, indexMatters=True, isCached=True, discBehavior=eHlpDiscBehavior.nothing ):
		attIndex = len(self.attributes)
		sys.stderr.write('creating attribute '+name+'\n')

		for a in self.attributes: # check uniqueness of names
			if a.name == name :
				raise MAH_msCommandException('attribute with the name [ '+name+' ] already exists')
			if a.shortName == shortName :
				raise MAH_msCommandException('attribute with the short name [ '+shortName+' ] already exists')

		fn = None
		if type == eHlpT.double or type == eHlpT.int or type == eHlpT.bool :
			fn = OpenMaya.MFnNumericAttribute()
		elif type == eHlpT.curve or type == eHlpT.mesh :
			fn = OpenMaya.MFnTypedAttribute()
		elif type == eHlpT.time :
			fn = OpenMaya.MFnUnitAttribute()
		elif type == eHlpT.ramp :
			fn = OpenMaya.MRampAttribute
		else:
			raise MAH_msCommandException('unsupported type to create attribute')

		mayaType = aHlpMayaTypes[type]

		mObject = None
		if type == eHlpT.ramp :
			mObject = fn.createCurveRamp( name, shortName)
		elif default is not None:
			mObject = fn.create( name, shortName, mayaType, default )
		else:
			mObject = fn.create( name, shortName, mayaType )

		if type != eHlpT.ramp :
			fn.setKeyable(isInput)
			fn.setWritable(isInput)
			fn.setReadable(not isInput)
			fn.setArray(isArray)
			if isArray and not isInput:
				raise MAH_msCommandException('Output Array attribute are not supported')
			fn.setCached(isCached)
			fn.setDisconnectBehavior( aHlpMayaDiscBehavior[discBehavior] )
			if isArray :
				fn.setIndexMatters(indexMatters)

		att = attDef(attIndex, mObject, name, shortName, type,  addAttr, isInput,isArray )
		
		# find the compute Group or create one
		computeGrpToUse = None
		listCpGpIds = computeGpIds
		if not isinstance(listCpGpIds, list): # if the variable is not a list then make it a list
			listCpGpIds = [computeGpIds]
		else :
			if not isInput:
				raise MAH_msCommandException('attribute '+name+' cannot be in multiple compute group and be an output, error')

		for CpGpId in listCpGpIds :
			if self.computeGroups[CpGpId] == None:
				computeGrpToUse = computeGroup(CpGpId)
				self.computeGroups[CpGpId] = computeGrpToUse
			else:
				computeGrpToUse = self.computeGroups[CpGpId]

			if isInput:
				computeGrpToUse.addIN( att )
			else:
				computeGrpToUse.addOUT( att )

		self.attributes.append( att )
		setattr(self.nodeClass, name, mObject )
		setattr(self.nodeClass, name+'Index', attIndex ) # maybe?

	def recursiveDeclareAffect( self, computeGp ):
		globalInList = []
		for cpGpDependent in computeGp.dependencies :
			attInSet = self.recursiveDeclareAffect( cpGpDependent )
			globalInList = set().union( globalInList , attInSet )
		
		globalInList = set().union( globalInList , computeGp.INS )
		# DO affects
		for attOut in computeGp.OUTS :
			for attIn in globalInList :
				sys.stderr.write(attIn.name+" affect "+attOut.name+"\n")
				self.nodeClass.attributeAffects( attIn.mObject, attOut.mObject )
		return globalInList

	def finalizeAttributeInitialization( self ):
		#---- first check the compute groups, if one has no OUT then create one "dummy" out for it
		sys.stderr.write('adding dummies output attributes\n')
		for cpGp in self.computeGroups:
			if len(cpGp.OUTS) == 0:
				dumName = 'dummyOUT_'+cpGp.INS[0].name
				numDum = 0
				if dumName in self.dummyNames:
					numDum = self.dummyNames[dumName] + 1
				self.dummyNames[dumName] = numDum
				dumName += str(numDum)
				self.createAtt( dumName, 'dO_'+cpGp.INS[0].shortName+str(numDum), eHlpT.int , cpGp.id, False )
		#---- then add all Attributes that need to be added to the node
		sys.stderr.write('adding attribute blueprints to node \n')
		for a in self.attributes:
			if not a.addAttr: # some attributes don't need to be added like double3
				continue
			self.nodeClass.addAttribute( a.mObject )
		#---- then we can declare affects
		sys.stderr.write('declare affects of attributes\n')
		for cpGp in self.computeGroups:
			if len(cpGp.affect) > 0: # this compute group affect something it is not a top affect group, skip
				continue
			self.recursiveDeclareAffect( cpGp )

	def addGroupDependencies( self, computeGpId,  computeGpDependencies ):
		if computeGpId not in range(len(self.computeGroups)) :
			raise MAH_msCommandException('compute group [ '+computeGpId+' ] does not exist')
		for cpGpDepId in computeGpDependencies :
			if cpGpDepId not in range(len(self.computeGroups)) :
				raise MAH_msCommandException('compute group [ '+cpGpDepId+' ] in dependency of '+computeGpId+' does not exist')
		self.computeGroups[computeGpId].addDependencies( [ self.computeGroups[cpGpId] for cpGpId in computeGpDependencies ] )

	def getArrayDataHdlVal( self, type, arrayDHDL ):
		if type == eHlpT.double :
			return arrayDHDL.asDouble()
		if type == eHlpT.bool :
			return arrayDHDL.asBool()
		if type == eHlpT.time :
			return arrayDHDL.asTime().value()
		if type == eHlpT.int :
			return arrayDHDL.asInt()
		if type == eHlpT.curve :
			return arrayDHDL.asNurbsCurveTransformed()
		raise MAH_msCommandException('type not handled in getAttValueOrHdl!')

	def getAttValueOrHdl( self, attId, thisNode, data ):
		if type(attId) != int:
			raise MAH_msCommandException('getAttValueOrHdl an take attribute Index as parameter')

		attDef = self.attributes[attId]

		if not attDef.isInput:
			return data.outputValue( attDef.mObject )

		if attDef.type == eHlpT.ramp :
			RampPlug = OpenMaya.MPlug( thisNode, attDef.mObject )
			RampPlug.setAttribute(attDef.mObject)
			RampPlug01 = OpenMaya.MPlug(RampPlug.node(), RampPlug.attribute())
			return OpenMaya.MRampAttribute(RampPlug01.node(), RampPlug01.attribute())
		
		tempData = None
		if attDef.isArray :
			tempData = data.inputArrayValue(attDef.mObject)
			elemCount = tempData.elementCount()
			result = []
			for idx in range(elemCount) :
				tempData.jumpToArrayElement(idx)
				result.append( self.getArrayDataHdlVal( attDef.type, tempData.inputValue() ) )
			return result
		else:
			tempData = data.inputValue(attDef.mObject)
			return self.getArrayDataHdlVal( attDef.type, tempData)

		#if attDef.type is "vectorOfDouble" :
		#	return OpenMaya.MVector( tempData.asVector() )

# class to help having a shorter/cleaner to read access to node's attribute values during the compute function
class ATTAccessor:
	def __init__( self, helper, node, data ):
		self.helper = helper
		self.node = node
		self.data = data

	def needCompute(self, cpGpId, plug):
		return self.helper.computeGroups[cpGpId].checkForPlug( self.node, plug )

	def setClean( self, cpGpId, data ):
		self.helper.computeGroups[cpGpId].setClean( data )

	def forceEvaluate( self, cpGpId ):
		self.helper.computeGroups[cpGpId].forceEvaluateOUTS( self.node )

	def get( self, att ):
		return self.helper.getAttValueOrHdl( att, self.node, self.data )

	def getIf( self, attTest, att ):
		boolVal = self.get( attTest )
		if boolVal:
			return self.get( att )

#  END Helper classes
#------------------------------------------------------------------------------- 


#----------------------------------------------------------------------------
# AEtemplate proc embedded in the plugin
def loadAETemplate():
	mel.eval(
		''' 
		''') # removed and put into its own file because that trick doesn't work in Maya 2012
#  END AE Template
#------------------------------------------------------------------------------- 

import numpy as np
import inspect, os

kPluginNodeName = "lightningProcessor"
kPluginNodeId = OpenMaya.MTypeId(0x8700B)

class MLG_msCommandException(Exception):
    def __init__(self,message):
        self.message = '[MLG] '+message
    
    def __str__(self):
        return self.message

# Compute groups used by lightning Node
eCG = enum( 'main', 'detail', 'tubeSides',
			'radiusAB', 'childLengthAB', 'intensityAB', 'chaosDisplacementAB', 'elevationAB', 'elevationRandAB', 'childProbabilityAB',
			'chaosDisplacementABRoot', 'elevationABRoot', 'elevationRandABRoot', 'radiusABRoot', 'childLengthABRoot', 'intensityABRoot', 'childProbabilityABRoot',
			'max' )

# Function used to convert the Processor output into a Maya mesh structure
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

	name = 'lightningIntensity'
	meshFn.createColorSetDataMesh(name)
	meshFn.setVertexColors( MColors, MVertexIds )
	
	return outData

def loadGeneralLightningScript():
	# here we load also the processor python script wich should be in the same folder as this plugin
	# see http://stackoverflow.com/questions/50499/in-python-how-do-i-get-the-path-and-name-of-the-file-that-is-currently-executin
	currentPluginPath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
	sys.path.append(currentPluginPath)
	if sys.modules.has_key('lightningboltProcessor') :
		del(sys.modules['lightningboltProcessor'])
		print 'module "lightningboltProcessor" deleted'
	mainLightningModule = __import__( 'lightningboltProcessor' )
	print 'module "lightningboltProcessor" loaded'

	return mainLightningModule

def fillArrayFromRampAtt( array, ramp, clampValues=None ):
	inc = 1.0/(array.size-1)
	param = 0.0
	valAt_util = OpenMaya.MScriptUtil()
	valAt_util.createFromDouble(1.0)
	valAtPtr = valAt_util.asFloatPtr()
	for i in range(array.size):
		ramp.getValueAtPosition(param,valAtPtr)

		if clampValues is None:
			array[i] = valAt_util.getFloat(valAtPtr)
		else:
			minv, maxv = clampValues
			array[i] = min ( maxv, max(valAt_util.getFloat(valAtPtr),minv) )
		param=param+inc	

def setupABVFromRamp( arrays, ramp, clampValues=None):
	array, arrayRoot = arrays
	if array.size <1 :
		raise MLG_msCommandException('array size must be more than 1')
	fillArrayFromRampAtt( array, ramp, clampValues)
	#sys.stderr.write('array '+str(array)+'\n')

def setupABRootVFromRamp( arrays, ramp, clampValues=None):
	array, arrayRoot = arrays
	if arrayRoot.size <1 :
		raise MLG_msCommandException('array size must be more than 1')
	if ramp is None :
		arrayRoot[:] = array # COPY Normal -> Root
		return		
	fillArrayFromRampAtt( arrayRoot, ramp, clampValues)


def getPointListFromCurve( numPoints, fnCurve, lengthScale):
	lengthScale = max(0.0, min(1.0,lengthScale) ) # clamp lengthScale to avoid query parameter outside [0,1]

	startTemp = OpenMaya.MScriptUtil()
	startTemp.createFromDouble(0.0)
	startPtr = startTemp.asDoublePtr()

	endTemp = OpenMaya.MScriptUtil()
	endTemp.createFromDouble(0.0)
	endPtr = endTemp.asDoublePtr()

	fnCurve.getKnotDomain(startPtr, endPtr)
	paramStart = startTemp.getDouble(startPtr)
	paramEnd = endTemp.getDouble(endPtr)-0.000001
	
	step = lengthScale*(paramEnd - paramStart)/(numPoints-1)
	
	pt=OpenMaya.MPoint()
	param=paramStart
	res = []
	for i in range(numPoints):
		fnCurve.getPointAtParam(param, pt)
		res.append( [pt.x,pt.y,pt.z] )
		param += step
	return res#, lengthScale*fnCurve.length()


class LBProcNode(OpenMayaMPx.MPxNode):

	#----------------------------------------------------------------------------
	# let's have the LightningBolt Module here as a static
	LM = loadGeneralLightningScript()

	#----------------------------------------------------------------------------	
	# helpers to help manage attributes on a Maya node, all the helpers are "statics" methodes of the node	
	MHLP = None
	#----------------------------------------------------------------------------

	def __init__(self):
		OpenMayaMPx.MPxNode.__init__(self)
		self.LP = LBProcNode.LM.lightningBoltProcessor(mayaLightningMesher) # instance a processor for this node

	def compute(self, plug, data):
		thisNode = self.thisMObject()
		acc = ATTAccessor( LBProcNode.MHLP, thisNode, data )
	# -- Compute Detail
		if acc.needCompute( eCG.detail, plug ):
			detailValue = acc.get(LBProcNode.detailIndex)
			self.LP.setGlobalValue( LBProcNode.LM.eGLOBAL.detail, detailValue ) # ***** initialisation of the sizes of arrays in the Processor (all Along Path arrays depend on detail)
			acc.setClean(eCG.detail, data)
	# -- Compute Tube Sides
		if acc.needCompute( eCG.tubeSides, plug ):
			tubeSidesValue = acc.get(LBProcNode.tubeSidesIndex)
			self.LP.setGlobalValue( LBProcNode.LM.eGLOBAL.tubeSides, tubeSidesValue )
			acc.setClean(eCG.tubeSides, data)
	# -- Compute Along Branch Ramp : Radius
		elif acc.needCompute( eCG.radiusAB, plug ):
			RadiusABRamp = acc.get(LBProcNode.radiusAlongBranchIndex)	
			setupABVFromRamp( self.LP.getAPBR( LBProcNode.LM.eAPBR.radius ), RadiusABRamp )
			if not acc.get( LBProcNode.radiusRootOverrideIndex ): # if no Root override then copy
				setupABRootVFromRamp( self.LP.getAPBR( LBProcNode.LM.eAPBR.radius ), None )
			acc.setClean(eCG.radiusAB, data)
	# -- Compute Along Branch Ramp : Intensity
		elif acc.needCompute( eCG.intensityAB, plug ):
			IntensityABRamp = acc.get(LBProcNode.intensityAlongBranchIndex)	
			setupABVFromRamp( self.LP.getAPBR( LBProcNode.LM.eAPBR.intensity ), IntensityABRamp )
			if not acc.get( LBProcNode.intensityRootOverrideIndex ): # if no Root override then copy
				setupABRootVFromRamp( self.LP.getAPBR( LBProcNode.LM.eAPBR.intensity ), None )
			acc.setClean(eCG.intensityAB, data)
	# -- Compute Along Branch Ramp : Child Length
		elif acc.needCompute( eCG.childLengthAB, plug ):
			childLengthABRamp = acc.get(LBProcNode.childLengthAlongBranchIndex)	
			setupABVFromRamp( self.LP.getAPBR( LBProcNode.LM.eAPBR.length ), childLengthABRamp )
			if not acc.get( LBProcNode.lengthRootOverrideIndex ): # if no Root override then copy
				setupABRootVFromRamp( self.LP.getAPBR( LBProcNode.LM.eAPBR.length ), None )
			acc.setClean(eCG.childLengthAB, data)
	# -- Compute Along Branch Ramp : Chaos Offset
		elif acc.needCompute( eCG.chaosDisplacementAB, plug ):
			chaosDisplacementABRamp = acc.get(LBProcNode.chaosDisplacementAlongBranchIndex)	
			setupABVFromRamp( self.LP.getAPSPE( LBProcNode.LM.eAPSPE.chaosDisplacement ), chaosDisplacementABRamp )
			if not acc.get( LBProcNode.chaosDisplacementRootOverrideIndex ): # if no Root override then copy
				setupABRootVFromRamp( self.LP.getAPSPE( LBProcNode.LM.eAPSPE.chaosDisplacement ), None )
			acc.setClean(eCG.chaosDisplacementAB, data)
	# -- Compute Along Branch Ramp : Elevation
		elif acc.needCompute( eCG.elevationAB, plug ):
			elevationABRamp = acc.get(LBProcNode.elevationAlongBranchIndex)
			setupABVFromRamp( self.LP.getAPSPE( LBProcNode.LM.eAPSPE.elevation ), elevationABRamp )
			if not acc.get( LBProcNode.elevationRootOverrideIndex ): # if no Root override then copy
				setupABRootVFromRamp( self.LP.getAPSPE( LBProcNode.LM.eAPSPE.elevation ), None )
			acc.setClean(eCG.elevationAB, data)
	# -- Compute Along Branch Ramp : Elevation Rand
		elif acc.needCompute( eCG.elevationRandAB, plug ):
			elevationRandABRamp = acc.get(LBProcNode.elevationRandAlongBranchIndex)	
			setupABVFromRamp( self.LP.getAPSPE( LBProcNode.LM.eAPSPE.elevationRand ), elevationRandABRamp )
			if not acc.get( LBProcNode.elevationRootOverrideIndex ): # if no Root override then copy
				setupABRootVFromRamp( self.LP.getAPSPE( LBProcNode.LM.eAPSPE.elevationRand ), None )
			acc.setClean(eCG.elevationRandAB, data)
	# -- Compute Along Branch Ramp : Child Probability
		elif acc.needCompute( eCG.childProbabilityAB, plug ):
			childProbabilityABRamp = acc.get(LBProcNode.childProbabilityAlongBranchIndex)
			# we clamp the values for childProbability, it's important since it remap array indices
			setupABVFromRamp( self.LP.getAPSPE( LBProcNode.LM.eAPSPE.childProbability ), childProbabilityABRamp, (0.0,1.0) )
			if not acc.get( LBProcNode.childrenNumberRootOverrideIndex ):# if no Root override then copy
				setupABRootVFromRamp( self.LP.getAPSPE( LBProcNode.LM.eAPSPE.childProbability ), None )			
			#setupABRootVFromRamp( self.LP.getAPSPE( LBProcNode.LM.eAPSPE.childProbability ), None ) # child probability need a copy of the normal values to the root because there is no override to do it
			acc.setClean(eCG.childProbabilityAB, data)
	# -- Compute ROOT Along Branch Ramp : Radius
		elif acc.needCompute( eCG.radiusABRoot, plug ):
			radiusABRootRamp = acc.get(LBProcNode.radiusAlongBranchRootIndex)	
			if not acc.get( LBProcNode.radiusRootOverrideIndex ):
				radiusABRootRamp = None # if none the values of the Normal Ramp are copied to the Root Ramp
			setupABRootVFromRamp( self.LP.getAPBR( LBProcNode.LM.eAPBR.radius ), radiusABRootRamp )
			acc.setClean(eCG.radiusABRoot, data)
	# -- Compute ROOT Along Branch Ramp : Child Length
		elif acc.needCompute( eCG.childLengthABRoot, plug ):
			childLengthABRootRamp = acc.get(LBProcNode.childLengthAlongBranchRootIndex)	
			if not acc.get( LBProcNode.lengthRootOverrideIndex ):
				childLengthABRootRamp = None # if none the values of the Normal Ramp are copied to the Root Ramp
			setupABRootVFromRamp( self.LP.getAPBR( LBProcNode.LM.eAPBR.length ), childLengthABRootRamp )
			acc.setClean(eCG.childLengthABRoot, data)
	# -- Compute ROOT Along Branch Ramp : Intensity
		elif acc.needCompute( eCG.intensityABRoot, plug ):
			intensityABRootRamp = acc.get(LBProcNode.intensityAlongBranchRootIndex)	
			if not acc.get( LBProcNode.intensityRootOverrideIndex ):
				intensityABRootRamp = None # if none the values of the Normal Ramp are copied to the Root Ramp
			setupABRootVFromRamp( self.LP.getAPBR( LBProcNode.LM.eAPBR.intensity ), intensityABRootRamp )
			acc.setClean(eCG.intensityABRoot, data)
	# -- Compute ROOT Along Branch Ramp : Chaos Offset
		elif acc.needCompute( eCG.chaosDisplacementABRoot, plug ):
			chaosDisplacementABRootRamp = acc.get(LBProcNode.chaosDisplacementAlongBranchRootIndex)	
			if not acc.get( LBProcNode.chaosDisplacementRootOverrideIndex ):
				chaosDisplacementABRootRamp = None # if none the values of the Normal Ramp are copied to the Root Ramp
			setupABRootVFromRamp( self.LP.getAPSPE( LBProcNode.LM.eAPSPE.chaosDisplacement ), chaosDisplacementABRootRamp )
			acc.setClean(eCG.chaosDisplacementABRoot, data)
	# -- Compute ROOT Along Branch Ramp : Elevation
		elif acc.needCompute( eCG.elevationABRoot, plug ):
			elevationABRootRamp = acc.get(LBProcNode.elevationAlongBranchRootIndex)	
			if not acc.get( LBProcNode.elevationRootOverrideIndex ):
				elevationABRootRamp = None
			setupABRootVFromRamp( self.LP.getAPSPE( LBProcNode.LM.eAPSPE.elevation ), elevationABRootRamp )
			acc.setClean(eCG.elevationABRoot, data)
	# -- Compute ROOT Along Branch Ramp : Elevation Rand
		elif acc.needCompute( eCG.elevationRandABRoot, plug ):
			elevationRandABRootRamp = acc.get(LBProcNode.elevationRandAlongBranchRootIndex)	
			if not acc.get( LBProcNode.elevationRootOverrideIndex ):
				elevationRandABRootRamp = None
			setupABRootVFromRamp( self.LP.getAPSPE( LBProcNode.LM.eAPSPE.elevationRand ), elevationRandABRootRamp )
			acc.setClean(eCG.elevationRandABRoot, data)
	# -- Compute ROOT Along Branch Ramp : Child Probability
		elif acc.needCompute( eCG.childProbabilityABRoot, plug ):
			childProbabilityABRootRamp = acc.get(LBProcNode.childProbabilityAlongBranchRootIndex)	
			if not acc.get( LBProcNode.childrenNumberRootOverrideIndex ):
				childProbabilityABRootRamp = None
			# we clamp the values for childProbability, it's important since it remap array indices
			setupABRootVFromRamp( self.LP.getAPSPE( LBProcNode.LM.eAPSPE.childProbability ), childProbabilityABRootRamp, (0.0,1.0) )
			acc.setClean(eCG.childProbabilityABRoot, data)

	# -- Compute MAIN
		elif acc.needCompute( eCG.main, plug ):
		#-------- Read All the attributes from plugs
			# global values
			timeValue = acc.get( LBProcNode.timeIndex)
			doAccumulateTimeValue = acc.get(LBProcNode.doAccumulateTimeIndex)
			startTimeAccumulationValue = acc.get(LBProcNode.startTimeAccumulationIndex)
			maxGenerationValue = acc.get( LBProcNode.maxGenerationIndex)
			vibrationFreqFactorValue = acc.get( LBProcNode.vibrationFreqFactorIndex)
			secondaryChaosFreqFactorValue = acc.get( LBProcNode.secondaryChaosFreqFactorIndex )
			secondaryChaosMinClampValue = acc.get( LBProcNode.secondaryChaosMinClampIndex )
			secondaryChaosMaxClampValue = acc.get( LBProcNode.secondaryChaosMaxClampIndex )
			secondaryChaosMinRemapValue = acc.get( LBProcNode.secondaryChaosMinRemapIndex )
			secondaryChaosMaxRemapValue = acc.get( LBProcNode.secondaryChaosMaxRemapIndex )
			seedChaosValue = acc.get( LBProcNode.seedChaosIndex)
			seedSkeletonValue = acc.get( LBProcNode.seedSkeletonIndex)

			# Along Branch dependent values
			radiusValue = acc.get( LBProcNode.radiusIndex)
			intensityValue = acc.get( LBProcNode.intensityIndex)
			lengthValue = acc.get( LBProcNode.lengthIndex)

			# Generation dependent values
			chaosFrequencyValue = acc.get( LBProcNode.chaosFrequencyIndex)
			chaosVibrationValue = acc.get( LBProcNode.chaosVibrationIndex)
			childrenNumberValue = acc.get( LBProcNode.childrenNumberIndex)
			childrenNumberRandValue = acc.get( LBProcNode.childrenNumberRandIndex)
			chaosTimeMultiplierValue = acc.get( LBProcNode.chaosTimeMultiplierIndex)
			skeletonTimeMultiplierValue = acc.get( LBProcNode.skeletonTimeMultiplierIndex)
			chaosDisplacementMultValue = acc.get( LBProcNode.chaosDisplacementIndex)
			lengthRandValue = acc.get( LBProcNode.lengthRandIndex)
			
			# Transfer of Along Branch dependent values
			transferRadiusValue = acc.get( LBProcNode.transferRadiusIndex)
			transferChildLengthValue = acc.get( LBProcNode.transferLengthIndex)
			transferIntensityValue = acc.get( LBProcNode.transferIntensityIndex)

			# Transfer of Generation dependent values
			transferChaosOffsetValue = acc.get( LBProcNode.transferChaosOffsetIndex)
			transferSkeletonTimeValue = acc.get( LBProcNode.transferSkeletonTimeIndex)
			transferChaosTimeValue = acc.get( LBProcNode.transferChaosTimeIndex)
			transferNumChildrenValue = acc.get( LBProcNode.transferNumChildrenIndex)
			transferNumChildrenRandValue = acc.get( LBProcNode.transferNumChildrenRandIndex)
			transferChaosFrequencyValue = acc.get( LBProcNode.transferChaosFrequencyIndex)
			transferChaosVibrationValue = acc.get( LBProcNode.transferChaosVibrationIndex)
			transferLengthRandValue = acc.get( LBProcNode.transferLengthRandIndex)

			# And the Root overrides that are not Ramp
			# Radius
			transferRadiusRootValue = acc.getIf( LBProcNode.radiusRootOverrideIndex, LBProcNode.transferRadiusRootIndex )
			# Childlength
			transferChildLengthRootValue = acc.getIf( LBProcNode.lengthRootOverrideIndex, LBProcNode.transferLengthRootIndex )
			# Intensity
			transferIntensityRootValue = acc.getIf( LBProcNode.intensityRootOverrideIndex, LBProcNode.transferIntensityRootIndex )
			# Times
			transferChaosTimeRootValue = acc.getIf( LBProcNode.timeRootOverrideIndex, LBProcNode.transferChaosTimeRootIndex )
			transferSkeletonTimeRootValue = acc.getIf( LBProcNode.timeRootOverrideIndex, LBProcNode.transferSkeletonTimeRootIndex )
			# Chaos
			transferChaosOffsetRootValue = acc.getIf( LBProcNode.chaosDisplacementRootOverrideIndex, LBProcNode.transferChaosOffsetRootIndex )
			transferChaosFrequencyRootValue = acc.getIf( LBProcNode.chaosDisplacementRootOverrideIndex, LBProcNode.transferChaosFrequencyRootIndex )
			# num Children
			transferNumChildrenRootValue = acc.getIf( LBProcNode.childrenNumberRootOverrideIndex, LBProcNode.transferNumChildrenRootIndex )
			transferNumChildrenRandRootValue = acc.getIf( LBProcNode.childrenNumberRootOverrideIndex, LBProcNode.transferNumChildrenRandRootIndex )

			tempCurves = acc.get(LBProcNode.inputCurvesIndex)
			if tempCurves == []:
				#sys.stderr.write('lightninbolt has no curve to compute!\n')
				return
			
			outputHandle = acc.get(LBProcNode.outputMeshIndex)

		#-------- END collect attributes from the node, now send everything to the lightning processor
			# load the list of points from the curve into the processor
			for c in tempCurves:
				fnNurbs = OpenMaya.MFnNurbsCurve( c )
				pointList = getPointListFromCurve( self.LP.maxPoints, fnNurbs, lengthValue )
				# load the list of points from the curve into the processor
				self.LP.addSeedPointList(pointList)

			# global values
			
			self.LP.setGlobalValue( LBProcNode.LM.eGLOBAL.time, timeValue )
			self.LP.setGlobalValue( LBProcNode.LM.eGLOBAL.doAccumulateTime, doAccumulateTimeValue )
			self.LP.setGlobalValue( LBProcNode.LM.eGLOBAL.startTimeAccumulation, startTimeAccumulationValue )
			self.LP.setGlobalValue( LBProcNode.LM.eGLOBAL.maxGeneration, maxGenerationValue )
			self.LP.setGlobalValue( LBProcNode.LM.eGLOBAL.vibrationFreqFactor, vibrationFreqFactorValue )
			self.LP.setGlobalValue( LBProcNode.LM.eGLOBAL.seedChaos, seedChaosValue )
			self.LP.setGlobalValue( LBProcNode.LM.eGLOBAL.seedSkeleton, seedSkeletonValue )
			self.LP.setGlobalValue( LBProcNode.LM.eGLOBAL.chaosSecondaryFreqFactor, secondaryChaosFreqFactorValue )
			self.LP.setGlobalValue( LBProcNode.LM.eGLOBAL.secondaryChaosMinClamp, secondaryChaosMinClampValue )
			self.LP.setGlobalValue( LBProcNode.LM.eGLOBAL.secondaryChaosMaxClamp, secondaryChaosMaxClampValue )
			self.LP.setGlobalValue( LBProcNode.LM.eGLOBAL.secondaryChaosMinRemap, secondaryChaosMinRemapValue )
			self.LP.setGlobalValue( LBProcNode.LM.eGLOBAL.secondaryChaosMaxRemap, secondaryChaosMaxRemapValue )

			# load the Along Path Values inputs of the node into the processor			
			self.LP.setAPVFactors( LBProcNode.LM.eAPBR.radius, radiusValue )
			self.LP.setAPVFactors( LBProcNode.LM.eAPBR.intensity, intensityValue )
			self.LP.setAPVFactors( LBProcNode.LM.eAPBR.length, lengthValue )
			# the corresponding transfer values
			self.LP.setAPVTransfer( LBProcNode.LM.eAPBR.radius, transferRadiusValue, transferRadiusRootValue )
			self.LP.setAPVTransfer( LBProcNode.LM.eAPBR.intensity, transferIntensityValue, transferIntensityRootValue )
			self.LP.setAPVTransfer( LBProcNode.LM.eAPBR.length, transferChildLengthValue, transferChildLengthRootValue )

			# load the Generation inputs
			self.LP.setGENValue( LBProcNode.LM.eGEN.chaosFrequency, chaosFrequencyValue )
			self.LP.setGENValue( LBProcNode.LM.eGEN.chaosVibration, chaosVibrationValue )
			self.LP.setGENValue( LBProcNode.LM.eGEN.chaosTime, chaosTimeMultiplierValue )
			self.LP.setGENValue( LBProcNode.LM.eGEN.skeletonTime, skeletonTimeMultiplierValue )
			self.LP.setGENValue( LBProcNode.LM.eGEN.childrenNumber, childrenNumberValue )
			self.LP.setGENValue( LBProcNode.LM.eGEN.childrenNumberRand, childrenNumberRandValue )
			self.LP.setGENValue( LBProcNode.LM.eGEN.chaosDisplacement, chaosDisplacementMultValue )
			self.LP.setGENValue( LBProcNode.LM.eGEN.lengthRand, lengthRandValue )
			# the corresponding transfer values
			self.LP.setGENTransfer( LBProcNode.LM.eGEN.skeletonTime, transferSkeletonTimeValue, transferSkeletonTimeRootValue )
			self.LP.setGENTransfer( LBProcNode.LM.eGEN.chaosTime, transferChaosTimeValue, transferChaosTimeRootValue )
			self.LP.setGENTransfer( LBProcNode.LM.eGEN.chaosDisplacement, transferChaosOffsetValue, transferChaosOffsetRootValue )
			self.LP.setGENTransfer( LBProcNode.LM.eGEN.childrenNumber, transferNumChildrenValue, transferNumChildrenRootValue )
			self.LP.setGENTransfer( LBProcNode.LM.eGEN.childrenNumberRand, transferNumChildrenRandValue, transferNumChildrenRandRootValue )
			self.LP.setGENTransfer( LBProcNode.LM.eGEN.chaosFrequency, transferChaosFrequencyValue, transferChaosFrequencyRootValue )
			self.LP.setGENTransfer( LBProcNode.LM.eGEN.chaosVibration, transferChaosVibrationValue )
			self.LP.setGENTransfer( LBProcNode.LM.eGEN.lengthRand, transferLengthRandValue )
								
			outputHandle.setMObject( self.LP.process() )
			acc.setClean(eCG.main, data)
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
		self.postConstructorRampInitialize( LBProcNode.radiusAlongBranch, [(0,1),(1,0)] )
		self.postConstructorRampInitialize( LBProcNode.radiusAlongBranchRoot, [(0,1),(1,0.5)] )
		self.postConstructorRampInitialize( LBProcNode.childLengthAlongBranch, [(0,1),(1,.25)] )
		self.postConstructorRampInitialize( LBProcNode.childLengthAlongBranchRoot, [(0,1),(1,.25)] )
		self.postConstructorRampInitialize( LBProcNode.intensityAlongBranch, [(0,1),(.15,0.8)] )
		self.postConstructorRampInitialize( LBProcNode.intensityAlongBranchRoot, [(0,1),(1,0.85)] )

		self.postConstructorRampInitialize( LBProcNode.chaosDisplacementAlongBranch, [(0,0),(.05,1)] )
		self.postConstructorRampInitialize( LBProcNode.chaosDisplacementAlongBranchRoot, [(0,0),(.05,1), (0.85,1.0), (1.0,.2)] )
		self.postConstructorRampInitialize( LBProcNode.elevationAlongBranch, [(0,0.12),(1,0)] )
		self.postConstructorRampInitialize( LBProcNode.elevationAlongBranchRoot, [(0,0.12),(1,0)] )
		self.postConstructorRampInitialize( LBProcNode.elevationRandAlongBranch, [(0,0.12),(0,0.3)] )
		self.postConstructorRampInitialize( LBProcNode.elevationRandAlongBranchRoot, [(0,0.12),(0,0.3)] )
		self.postConstructorRampInitialize( LBProcNode.childProbabilityAlongBranch, [(0,0),(1,1)] )
		self.postConstructorRampInitialize( LBProcNode.childProbabilityAlongBranchRoot, [(0,0),(1,1)] )
		

def nodeCreator():
	return OpenMayaMPx.asMPxPtr( LBProcNode() )

def nodeInitializer():
	LBProcNode.MHLP = MayaAttHelper(LBProcNode, eCG.max )

	LBProcNode.MHLP.createAtt( 'inputCurves', 'ic', eHlpT.curve, eCG.main, isArray=True, indexMatters=False, discBehavior=eHlpDiscBehavior.delete )

# Global Values
	LBProcNode.MHLP.createAtt( 'detail', 'det', eHlpT.int, eCG.detail, default=6 )
	LBProcNode.MHLP.createAtt( 'tubeSides', 'ts', eHlpT.int, eCG.tubeSides, default=4 )

	LBProcNode.MHLP.createAtt( 'time', 't', eHlpT.time, eCG.main, default=0.0 )
	LBProcNode.MHLP.createAtt( 'doAccumulateTime', 'dat', eHlpT.bool, eCG.main, default=False )
	LBProcNode.MHLP.createAtt( 'startTimeAccumulation', 'sta', eHlpT.time, eCG.main, default=1.0 )

	LBProcNode.MHLP.createAtt( 'maxGeneration', 'mg', eHlpT.int, eCG.main, default=3 )
	LBProcNode.MHLP.createAtt( 'seedChaos', 'sc', eHlpT.int, eCG.main, default=0 )
	LBProcNode.MHLP.createAtt( 'seedSkeleton', 'ss', eHlpT.int, eCG.main, default=0 )
	LBProcNode.MHLP.createAtt( 'secondaryChaosFreqFactor', 'scff', eHlpT.double, eCG.main, default=4.5 )
	LBProcNode.MHLP.createAtt( 'secondaryChaosMinClamp', 'scmnc', eHlpT.double, eCG.main, default=.25 )
	LBProcNode.MHLP.createAtt( 'secondaryChaosMaxClamp', 'scmxc', eHlpT.double, eCG.main, default=0.75 )
	LBProcNode.MHLP.createAtt( 'secondaryChaosMinRemap', 'scmnr', eHlpT.double, eCG.main, default=0.6 )
	LBProcNode.MHLP.createAtt( 'secondaryChaosMaxRemap', 'scmxr', eHlpT.double, eCG.main, default=1.0 )
	LBProcNode.MHLP.createAtt( 'vibrationFreqFactor', 'vff', eHlpT.double, eCG.main, default=6.0 )

# Generation Values
	LBProcNode.MHLP.createAtt( 'chaosTimeMultiplier', 'ctm', eHlpT.double, eCG.main, default=8.0 )
	LBProcNode.MHLP.createAtt( 'skeletonTimeMultiplier', 'stm', eHlpT.double, eCG.main, default=6.5 )
	LBProcNode.MHLP.createAtt( 'chaosFrequency', 'cf', eHlpT.double, eCG.main, default=0.15 )
	LBProcNode.MHLP.createAtt( 'chaosVibration', 'cv', eHlpT.double, eCG.main, default=0.15 )
	LBProcNode.MHLP.createAtt( 'childrenNumber', 'cn', eHlpT.int, eCG.main, default=1 )
	LBProcNode.MHLP.createAtt( 'childrenNumberRand', 'cnr', eHlpT.int, eCG.main, default=3 )
	LBProcNode.MHLP.createAtt( 'chaosDisplacement', 'cd', eHlpT.double, eCG.main, default=1.0 )
	LBProcNode.MHLP.createAtt( 'lengthRand', 'lr', eHlpT.double, eCG.main, default=0.5 )

	# Generation Transfer Factors
	LBProcNode.MHLP.createAtt( 'transferSkeletonTime', 'tst', eHlpT.double, eCG.main, default=3.0 )
	LBProcNode.MHLP.createAtt( 'transferChaosTime', 'tct', eHlpT.double, eCG.main, default=1.0 )
	LBProcNode.MHLP.createAtt( 'transferChaosFrequency', 'tcf', eHlpT.double, eCG.main, default=2.0 )
	LBProcNode.MHLP.createAtt( 'transferChaosVibration', 'tcv', eHlpT.double, eCG.main, default=.6 )
	LBProcNode.MHLP.createAtt( 'transferNumChildren', 'tnc', eHlpT.double, eCG.main, default=0.5 )
	LBProcNode.MHLP.createAtt( 'transferNumChildrenRand', 'tncr', eHlpT.double, eCG.main, default=0.75 )
	LBProcNode.MHLP.createAtt( 'transferChaosOffset', 'tco', eHlpT.double, eCG.main, default=0.6 )
	LBProcNode.MHLP.createAtt( 'transferLengthRand', 'tlr', eHlpT.double, eCG.main, default=1.5 )

# APV Branches
	LBProcNode.MHLP.createAtt( 'radiusAlongBranch', 'rab', eHlpT.ramp, eCG.radiusAB )
	LBProcNode.MHLP.createAtt( 'radius', 'r', eHlpT.double, eCG.main, default=.15 )
	LBProcNode.MHLP.createAtt( 'childLengthAlongBranch', 'clab', eHlpT.ramp, eCG.childLengthAB )
	LBProcNode.MHLP.createAtt( 'length', 'l', eHlpT.double, eCG.main, default=1.0 )
	LBProcNode.MHLP.createAtt( 'intensityAlongBranch', 'iab', eHlpT.ramp, eCG.intensityAB )
	LBProcNode.MHLP.createAtt( 'intensity', 'i', eHlpT.double, eCG.main, default=1.0 )

	# APV Branches Transfer Factors
	LBProcNode.MHLP.createAtt( 'transferRadius', 'tr', eHlpT.double, eCG.main, default=0.8 )
	LBProcNode.MHLP.createAtt( 'transferLength', 'tl', eHlpT.double, eCG.main, default=0.6 )
	LBProcNode.MHLP.createAtt( 'transferIntensity', 'ti', eHlpT.double, eCG.main, default=1.0 )

# APV Specials
	LBProcNode.MHLP.createAtt( 'chaosDisplacementAlongBranch', 'cdab', eHlpT.ramp, eCG.chaosDisplacementAB )

	# elevation 1 = 180 degree
	LBProcNode.MHLP.createAtt( 'elevationAlongBranch', 'eab', eHlpT.ramp, eCG.elevationAB )
	LBProcNode.MHLP.createAtt( 'elevationRandAlongBranch', 'erab', eHlpT.ramp, eCG.elevationRandAB )

	LBProcNode.MHLP.createAtt( 'childProbabilityAlongBranch', 'cpab', eHlpT.ramp, eCG.childProbabilityAB )

# Root Overrides
	# Generation Transfer Overrides
	LBProcNode.MHLP.createAtt( 'timeRootOverride', 'tro', eHlpT.bool, eCG.main, default=False )
	LBProcNode.MHLP.createAtt( 'transferChaosTimeRoot', 'tctrt', eHlpT.double, eCG.main, default=1.0 )
	LBProcNode.MHLP.createAtt( 'transferSkeletonTimeRoot', 'tstrt', eHlpT.double, eCG.main, default=1.0 )

	LBProcNode.MHLP.createAtt( 'chaosDisplacementRootOverride', 'coro', eHlpT.bool, eCG.chaosDisplacementABRoot, default=True )
	LBProcNode.MHLP.createAtt( 'chaosDisplacementAlongBranchRoot', 'orrt', eHlpT.ramp, eCG.chaosDisplacementABRoot )
	LBProcNode.MHLP.createAtt( 'transferChaosOffsetRoot', 'tor', eHlpT.double, eCG.main, default=0.55 )
	LBProcNode.MHLP.createAtt( 'transferChaosFrequencyRoot', 'tcfr', eHlpT.double, eCG.main, default=2.0 )

	LBProcNode.MHLP.createAtt( 'childrenNumberRootOverride', 'ncro', eHlpT.bool, eCG.main, default=0 )
	LBProcNode.MHLP.createAtt( 'transferNumChildrenRoot', 'tncrt', eHlpT.double, eCG.main, default=1.0 )
	LBProcNode.MHLP.createAtt( 'transferNumChildrenRandRoot', 'tncrrt', eHlpT.double, eCG.main, default=1.0 )
	LBProcNode.MHLP.createAtt( 'childProbabilityAlongBranchRoot', 'cpabr', eHlpT.ramp, eCG.childProbabilityABRoot )


	# APV Special overrides
	LBProcNode.MHLP.createAtt( 'elevationRootOverride', 'ero', eHlpT.bool, [eCG.elevationABRoot, eCG.elevationRandABRoot], default=False )
	LBProcNode.MHLP.createAtt( 'elevationAlongBranchRoot', 'errt', eHlpT.ramp, eCG.elevationABRoot )	
	LBProcNode.MHLP.createAtt( 'elevationRandAlongBranchRoot', 'errrt', eHlpT.ramp, eCG.elevationRandABRoot )

	# APV Branches overrides
	LBProcNode.MHLP.createAtt( 'radiusRootOverride', 'rro', eHlpT.bool, eCG.radiusABRoot, default=True )
	LBProcNode.MHLP.createAtt( 'radiusAlongBranchRoot', 'rrrt', eHlpT.ramp, eCG.radiusABRoot )
	LBProcNode.MHLP.createAtt( 'transferRadiusRoot', 'trr', eHlpT.double, eCG.main, default=0.45 )
	
	LBProcNode.MHLP.createAtt( 'lengthRootOverride', 'lro', eHlpT.bool, eCG.childLengthABRoot, default=True )
	LBProcNode.MHLP.createAtt( 'childLengthAlongBranchRoot', 'clapr', eHlpT.ramp, eCG.childLengthABRoot )
	LBProcNode.MHLP.createAtt( 'transferLengthRoot', 'tlro', eHlpT.double, eCG.main, default=0.4 )
	
	LBProcNode.MHLP.createAtt( 'intensityRootOverride', 'iro', eHlpT.bool, eCG.intensityABRoot, default=True )
	LBProcNode.MHLP.createAtt( 'intensityAlongBranchRoot', 'iabr', eHlpT.ramp, eCG.intensityABRoot )
	LBProcNode.MHLP.createAtt( 'transferIntensityRoot', 'tir', eHlpT.double, eCG.main, default=1.0 )

# OUTPUTS
	LBProcNode.MHLP.createAtt( 'outputMesh', 'om', eHlpT.mesh, eCG.main, isInput=False )

# Dependencies of some of the Compute Groups
	LBProcNode.MHLP.addGroupDependencies( eCG.radiusAB, [eCG.detail] )
	LBProcNode.MHLP.addGroupDependencies( eCG.childLengthAB, [eCG.detail] )
	LBProcNode.MHLP.addGroupDependencies( eCG.intensityAB, [eCG.detail] )
	LBProcNode.MHLP.addGroupDependencies( eCG.chaosDisplacementAB, [eCG.detail] )
	LBProcNode.MHLP.addGroupDependencies( eCG.elevationAB, [eCG.detail] )
	LBProcNode.MHLP.addGroupDependencies( eCG.elevationRandAB, [eCG.detail] )
	LBProcNode.MHLP.addGroupDependencies( eCG.childProbabilityAB, [eCG.detail] )
	
	LBProcNode.MHLP.addGroupDependencies( eCG.radiusABRoot, [eCG.detail] )
	LBProcNode.MHLP.addGroupDependencies( eCG.childLengthABRoot, [eCG.detail] )
	LBProcNode.MHLP.addGroupDependencies( eCG.intensityABRoot, [eCG.detail] )
	LBProcNode.MHLP.addGroupDependencies( eCG.chaosDisplacementABRoot, [eCG.detail] )
	LBProcNode.MHLP.addGroupDependencies( eCG.elevationABRoot, [eCG.detail] )
	LBProcNode.MHLP.addGroupDependencies( eCG.elevationRandABRoot, [eCG.detail] )
	LBProcNode.MHLP.addGroupDependencies( eCG.childProbabilityABRoot, [eCG.detail] )
	
	LBProcNode.MHLP.addGroupDependencies( eCG.main, [ eCG.tubeSides, eCG.radiusAB, eCG.childLengthAB, eCG.intensityAB, eCG.chaosDisplacementAB, eCG.elevationAB, eCG.elevationRandAB, eCG.childProbabilityAB, eCG.radiusABRoot, eCG.childLengthABRoot, eCG.intensityABRoot, eCG.chaosDisplacementABRoot, eCG.elevationABRoot, eCG.elevationRandABRoot, eCG.childProbabilityABRoot ] )

# Adding attributes, generating affects etc...
	LBProcNode.MHLP.finalizeAttributeInitialization()

def cleanupClass():
	LBProcNode.MHLP.cleanUp()
	delattr(LBProcNode,'LM')
	delattr(LBProcNode,'MHLP')

# initialize the script plug-in
def initializePlugin(mobject):
	#loadAETemplate()
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
