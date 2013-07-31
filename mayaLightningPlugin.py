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
- You need pymel installed
================================================================================
* USAGE:
- if you want to put the plugin into the Maya Folder, be sure to move this file and lightningboltProcessor.py together
- the AEtemplate is embedded in this file, so you don't need to install it
- load the plugin in Maya, if you didn't move the files in a Maya Folder you can just execute
pm.loadPlugin('{YOUR LIGHTNING PLUGIN PATH}/mayaLightningPlugin.py')
- select some curves and execute this in the python script Editor:

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

import numpy as np
import math

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

	mObj = meshFn.create(numVertices, numFaces, Mpoints , MfacesCount, MfaceConnects, outData)

# vertexColors
	scrUtil.createFromList( resultIntensityColors, len(resultIntensityColors))
	MColors = OpenMaya.MColorArray( scrUtil.asDouble4Ptr(), numVertices )

	scrUtil.createFromList( range(numVertices) , numVertices )
	MVertexIds = OpenMaya.MIntArray( scrUtil.asIntPtr(), numVertices )

	name = 'lightningIntensity'
	fName = meshFn.createColorSetDataMesh(name)
	meshFn.setVertexColors( MColors, MVertexIds )
	
	return outData

def loadGeneralLightningScript():
	# here we load also the processor python script wich should be in the same folder as this plugin
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
	# AEtemplate proc embedded in the plugin
	mel.eval('''

proc string localizedAttrName( string $name ) {
	if( $name == "Selected Color" ) {
		return (uiRes("m_AEaddRampControl.kSelClr"));
	} else if( $name == "Selected Position" ) {
		return (uiRes("m_AEaddRampControl.kSelPos"));
	} else if( $name == "Selected Value" ) {
		return (uiRes("m_AEaddRampControl.kSelVal"));
	} else if( $name == "Interpolation method" ) {
		return (uiRes("m_AEaddRampControl.kInterpMethod"));
	}
}

global proc LGHTAEmakeLargeRamp( string $nodeAttr,
							int $bound,
							int $indent,
							int $staticEntries,
							int $staticPositions,
							int $adaptiveScaling )
//
//	Description:
//
//	$staticEntries - If set to 1, the gradientControl widget will be set to
//					a fixed number of ramp entries (control points).
//
{
	string $buffer[];
	tokenize($nodeAttr, ".", $buffer);
	string $attr = $buffer[1];

	string $rampName = $attr + "Ramp";
	string $editButton = $attr + "RampEdit";
	string $scName = $attr +"Scc";
	string $spName = $attr +"Sp";
	string $siName = $attr +"Si";

	string $colEntryName =  ($nodeAttr + "[0]." + $attr + "_ColorR");
	int $isColor = `objExists ($colEntryName)`;

	//setUITemplate -pst attributeEditorTemplate;
	string $niceN = `attributeName -nice $nodeAttr`;
	columnLayout -rowSpacing 2 -cat "left" 15 -adj true;
	text -label $niceN -align "left";

	// ramp block
	string $rampForm = `formLayout ($rampName + "Form")`;
		string $spc	= `attrFieldSliderGrp -label (localizedAttrName("Selected Position"))
			-cw 1 123 -annotation (localizedAttrName("Selected Position")) $spName`;
		string $scc;
		if( $isColor ){
			$scc= `attrColorSliderGrp -label (localizedAttrName("Selected Color"))
				-cw 1 123 -cw 2 45 -cw 3 0 -annotation (localizedAttrName("Selected Color")) -sb 1 $scName`;
		} else {
			$scc	= `attrFieldSliderGrp -label (localizedAttrName("Selected Value"))
			-cw 1 123 -annotation (localizedAttrName("Selected Value")) $scName`;
		}
		
		string $interp = `attrEnumOptionMenuGrp -label (uiRes("m_AEaddRampControl.kInterp"))
			-cw 1 123 -annotation (localizedAttrName("Interpolation method")) $siName`;
		string $boundStyle = "etchedIn";
		if( $bound ){
			$boundStyle = "in";
		}
		string $lmax;
		if ( $adaptiveScaling ) {
			$lmax = `text -label "1.0" ($rampName+"LX")`;
		}
		$editButton = `button -l ">" -width 23 -c ("editRampAttribute "+$nodeAttr) $editButton`;
		string $rframe = `frameLayout -bs $boundStyle -lv 0 -cll 0 ($rampForm + "fr")`;
		string $widgetName = `gradientControl
								-at $nodeAttr
								-snc $staticEntries
								-sp $staticPositions
								// -w 148 -h 74
								-w 135 -h 74
								$rampName`;
		if ( $adaptiveScaling ) {
			gradientControl -e -as $adaptiveScaling -ror $adaptiveScaling -ulc $lmax $widgetName;
		}

		setParent ..;
		formLayout
			-edit
				-attachForm $spc "left"  0
				-attachNone $spc "right"
				-attachForm $spc "top" 0
				-attachNone $spc "bottom"

				-attachForm $scc "left" 0 
				-attachNone $scc "right"
				-attachControl $scc "top" 0 $spc
				-attachNone $scc "bottom"

				-attachForm $interp "left" 0 
				-attachNone $interp "right"
				-attachControl $interp "top" 0 $scc
				-attachNone $interp "bottom"

				-attachControl $rframe "left" 2 $interp
				-attachNone $rframe "right"
				-attachForm $rframe "top" 0
				-attachNone $rframe "bottom"

				-attachForm $editButton "top" 0
				-attachForm $editButton "bottom" 0
				-attachControl $editButton "left" 25 $rframe
				-attachNone $editButton "right"
				$rampForm;

		if ( $adaptiveScaling ) {
			formLayout
				-edit
					-attachControl $lmax "left" 2 $rframe
					-attachNone $lmax "right"
					-attachForm $lmax "top" 0
					-attachNone $lmax "bottom"
					$rampForm;
		}
	setParent ..;
	setParent ..;

	// input
	if(objExists ($nodeAttr +"Input")){
		string $inLabel;
		string $labelAttr = `attributeName -nice $nodeAttr`;
		string $inputVarAnnot = (uiRes("m_AEaddRampControl.kInputVarAnnot"));
		
		if( $indent || size( $labelAttr ) < 9 ){
			string $fmt = (uiRes("m_AEaddRampControl.kInputFmt"));
			$inLabel = `format -s $labelAttr $fmt`;
		} else {
			$inLabel = (uiRes("m_AEaddRampControl.kInputShort"));
		}
		if( $indent ){
			attrEnumOptionMenuGrp -l $inLabel
				-ann $inputVarAnnot
				-cw 1 204
				-at ($nodeAttr + "Input") ($rampName + "Input");
		} else {
			attrEnumOptionMenuGrp -l $inLabel
				-ann $inputVarAnnot
				-cw 1 123
				-at ($nodeAttr + "Input") ($rampName + "Input");
		}
		
	}

	// inputBias
	if(  objExists ($nodeAttr +"InputBias") ){
		attrFieldSliderGrp -label (uiRes("m_AEaddRampControl.kInputBias")) -cw4 123 81 130	25 
				-at ($nodeAttr +"InputBias") ($rampName + "InputBias");
	}

	// inputScale
	if(  objExists ($nodeAttr +"InputScale") ){
		attrFieldSliderGrp -label (uiRes("m_AEaddRampControl.kInputScale")) -cw4 123 81 130 25	
				-at ($nodeAttr +"InputScale") ($rampName + "InputScale");
	}
	// inputMax
	if(  objExists ($nodeAttr +"InputMax") ){
		attrFieldSliderGrp -label (uiRes("m_AEaddRampControl.kInputMax")) -cw4 123 81 130 25	
				-at ($nodeAttr +"InputMax") ($rampName + "InputMax");
	}
	// inputOffset
	if(  objExists ($nodeAttr +"InputOffset") ){
		attrFieldSliderGrp -label (uiRes("m_AEaddRampControl.kInputOffset")) -cw4 123 81 130 25	
				-at ($nodeAttr +"InputOffset") ($rampName + "InputOffset");
	}

	// tell the port about the controls
	gradientControl -e -scc $scc $widgetName;
	gradientControl -e -spc $spc $widgetName;
	gradientControl -e -sic $interp $widgetName;
	//setUITemplate -ppt;

}

global proc LGHTAERampControl(string $nodeAttr)
{
	LGHTAEmakeLargeRamp( $nodeAttr,1,1,0,0, 0 );
}

global proc LGHTAERampControl_Replace (string $nodeAttr)
{
	string $buffer[];
	tokenize($nodeAttr, ".", $buffer);
	string $attr = $buffer[1];

	string $rampName = $attr + "Ramp";
	string $editButton = $attr + "RampEdit";

	gradientControl -edit -at $nodeAttr $rampName;
	if( `button -exists $editButton` ){	
		button -edit -c ("editRampAttribute "+$nodeAttr) $editButton;
	}
	if( `objExists ($nodeAttr +"Input")` ){
		attrEnumOptionMenuGrp -edit -at ($nodeAttr + "Input") ($rampName + "Input");
	}
	if( `objExists ($nodeAttr +"InputScale")` ){
		attrFieldSliderGrp -edit -at ($nodeAttr + "InputScale") ($rampName + "InputScale");
	}
	if( `objExists ($nodeAttr +"InputBias")` ){
		attrFieldSliderGrp -edit -at ($nodeAttr + "InputBias") ($rampName + "InputBias");
	}
	if( `objExists ($nodeAttr +"InputMax")` ){
		attrFieldSliderGrp -edit -at ($nodeAttr + "InputMax") ($rampName + "InputMax");
	}
	if( `objExists ($nodeAttr +"InputOffset")` ){
		if (`attrFieldSliderGrp -query -exists ($rampName + "InputOffset")`) {
			attrFieldSliderGrp -edit -at ($nodeAttr + "InputOffset") ($rampName + "InputOffset");
		}
	}
}

global proc AElightningProcessorTemplate( string $nodeName )
{
	AEswatchDisplay  $nodeName;
	editorTemplate -beginScrollLayout;

		editorTemplate -beginLayout "System parameters" -collapse 1;
			editorTemplate -addControl "time";
			editorTemplate -callCustom "LGHTAETimeAccumulateLayout" "LGHTAETimeAccumulateLayout_Replace" "doAccumulateTime" "startTimeAccumulation";
			editorTemplate -addControl "maxGeneration";
			editorTemplate -addControl "seedChaos";
			editorTemplate -addControl "seedSkeleton";
			editorTemplate -addControl "tubeSides";
			editorTemplate -addControl "detail";
			editorTemplate -s "doAccumulateTimeValue";
			editorTemplate -s "startTimeAccumulation";
		editorTemplate -endLayout;
		
		editorTemplate -beginLayout "⟦ Time Attributes ⟧" -collapse 1;
			editorTemplate -callCustom "LGHTAELineBaseFloatTransfertLayout" "LGHTAELineBaseTransfertLayout_Replace" "chaosTimeMultiplier" "transfertChaosTime";
			editorTemplate -callCustom "LGHTAELineBaseFloatTransfertLayout" "LGHTAELineBaseTransfertLayout_Replace" "skeletonTimeMultiplier" "transfertSkeletonTime";
			editorTemplate -callCustom "LGHTAEOverrideLayoutTime" "LGHTAEOverrideLayoutTime_Replace" "timeRootOverride" "transfertChaosTimeRoot" "transfertSkeletonTimeRoot";				
			editorTemplate -s "chaosTimeMultiplier";
			editorTemplate -s "vibrationFreqFactor";
			editorTemplate -s "transfertChaosTime";
			editorTemplate -s "skeletonTimeMultiplier";
			editorTemplate -s "transfertSkeletonTime";
			editorTemplate -s "timeRootOverride";
			editorTemplate -s "transfertChaosTimeRoot";
			editorTemplate -s "transfertSkeletonTimeRoot";	
		editorTemplate -endLayout;

		editorTemplate -beginLayout "⟦ Length Attributes ⟧" -collapse 1;
			editorTemplate -callCustom "LGHTAELineBaseFloatTransfertLayout" "LGHTAELineBaseTransfertLayout_Replace" "length" "transfertLength";
			editorTemplate -callCustom "LGHTAELineBaseFloatTransfertLayout" "LGHTAELineBaseTransfertLayout_Replace" "lengthRand" "transfertLengthRand";
			editorTemplate -callCustom "LGHTAERampControl" "LGHTAERampControl_Replace" "childLengthAlongBranch";
			editorTemplate -callCustom "LGHTAEOverrideLayoutLength" "LGHTAEOverrideLayoutLength_Replace" "lengthRootOverride" "transfertLengthRoot" "childLengthAlongBranchRoot";

			editorTemplate -s "length";
			editorTemplate -s "transfertLength";
			editorTemplate -s "lengthRand";
			editorTemplate -s "transfertLengthRand";
			editorTemplate -s "childLengthAlongBranch";
			editorTemplate -s "lengthRootOverride";
			editorTemplate -s "transfertLengthRoot";
			editorTemplate -s "childLengthAlongBranchRoot";
		editorTemplate -endLayout;

		editorTemplate -beginLayout "⟦ Radius Attributes ⟧" -collapse 1;
			editorTemplate -callCustom "LGHTAELineBaseFloatTransfertLayout" "LGHTAELineBaseTransfertLayout_Replace" "radius" "transfertRadius";
			editorTemplate -callCustom "LGHTAERampControl" "LGHTAERampControl_Replace" "radiusAlongBranch";
			editorTemplate -callCustom "LGHTAEOverrideLayoutRadius" "LGHTAEOverrideLayoutRadius_Replace" "radiusRootOverride" "transfertRadiusRoot" "radiusAlongBranchRoot";
			editorTemplate -s "radius";
			editorTemplate -s "transfertRadius";
			editorTemplate -s "radiusAlongBranch";
			editorTemplate -s "radiusRootOverride";
			editorTemplate -s "transfertRadiusRoot";
			editorTemplate -s "radiusAlongBranchRoot";
		editorTemplate -endLayout;

		editorTemplate -beginLayout "⟦ Intensity Attributes ⟧ (Vertex Color)" -collapse 1;
			editorTemplate -callCustom "LGHTAELineBaseFloatTransfertLayout" "LGHTAELineBaseTransfertLayout_Replace" "intensity" "transfertIntensity";
			editorTemplate -callCustom "LGHTAERampControl" "LGHTAERampControl_Replace" "intensityAlongBranch";
			editorTemplate -callCustom "LGHTAEOverrideLayoutIntensity" "LGHTAEOverrideLayoutIntensity_Replace" "intensityRootOverride" "transfertIntensityRoot" "intensityAlongBranchRoot";

			editorTemplate -s "intensity";
			editorTemplate -s "transfertIntensity";
			editorTemplate -s "intensityAlongBranch";
			editorTemplate -s "intensityRootOverride";
			editorTemplate -s "intensityAlongBranchRoot";
			editorTemplate -s "transfertIntensityRoot";	
		editorTemplate -endLayout;
	
		editorTemplate -beginLayout "⟦ Chaos Attributes ⟧" -collapse 1;
			editorTemplate -callCustom "LGHTAELineBaseFloatTransfertLayout" "LGHTAELineBaseTransfertLayout_Replace" "chaosDisplacement" "transfertChaosOffset";
			editorTemplate -callCustom "LGHTAELineBaseFloatTransfertLayout" "LGHTAELineBaseTransfertLayout_Replace" "chaosFrequency" "transfertChaosFrequency";
			editorTemplate -callCustom "LGHTAERampControl" "LGHTAERampControl_Replace" "chaosDisplacementAlongBranch";
			editorTemplate -callCustom "LGHTAEOverrideLayoutChaos" "LGHTAEOverrideLayoutChaos_Replace" "chaosDisplacementRootOverride" "transfertChaosOffsetRoot" "transfertChaosFrequencyRoot" "chaosDisplacementAlongBranchRoot";

			editorTemplate -callCustom "LGHTAELineBaseFloatAttribute" "LGHTAELineBaseAttribute_Replace" "secondaryChaosFreqFactor";
			editorTemplate -callCustom "LGHTAELineBaseFloatAttribute" "LGHTAELineBaseAttribute_Replace" "secondaryChaosMinClamp";
			editorTemplate -callCustom "LGHTAELineBaseFloatAttribute" "LGHTAELineBaseAttribute_Replace" "secondaryChaosMaxClamp";
			editorTemplate -callCustom "LGHTAELineBaseFloatAttribute" "LGHTAELineBaseAttribute_Replace" "secondaryChaosMinRemap";
			editorTemplate -callCustom "LGHTAELineBaseFloatAttribute" "LGHTAELineBaseAttribute_Replace" "secondaryChaosMaxRemap";
			editorTemplate -callCustom "LGHTAELineBaseFloatTransfertLayout" "LGHTAELineBaseTransfertLayout_Replace" "chaosVibration" "transfertChaosVibration";
			editorTemplate -callCustom "LGHTAELineBaseFloatAttribute" "LGHTAELineBaseAttribute_Replace" "vibrationFreqFactor";

			editorTemplate -s "chaosDisplacementRootOverride";
			editorTemplate -s "transfertChaosOffsetRoot";
			editorTemplate -s "chaosDisplacementAlongBranchRoot";
			editorTemplate -s "transfertChaosFrequencyRoot";

			editorTemplate -s "chaosDisplacement";
			editorTemplate -s "transfertChaosOffset";
			editorTemplate -s "chaosFrequency";
			editorTemplate -s "transfertChaosFrequency";
			editorTemplate -s "chaosVibration";
			editorTemplate -s "transfertChaosVibration";

			editorTemplate -s "secondaryChaosFreqFactor";
			editorTemplate -s "secondaryChaosMinClamp";
			editorTemplate -s "secondaryChaosMaxClamp";
			editorTemplate -s "secondaryChaosMinRemap";
			editorTemplate -s "secondaryChaosMaxRemap";
			editorTemplate -s "vibrationFreqFactor";

			editorTemplate -s "chaosDisplacementAlongBranch";
		editorTemplate -endLayout;

		editorTemplate -beginLayout "⟦ Childs Generation Attributes ⟧" -collapse 1;
			editorTemplate -callCustom "LGHTAELineBaseIntTransfertLayout" "LGHTAELineBaseTransfertLayout_Replace" "childrenNumber" "transfertNumChildren";
			editorTemplate -callCustom "LGHTAELineBaseIntTransfertLayout" "LGHTAELineBaseTransfertLayout_Replace" "childrenNumberRand" "transfertNumChildrenRand";
			editorTemplate -callCustom "LGHTAERampControl" "LGHTAERampControl_Replace" "childProbabilityAlongBranch";
			editorTemplate -callCustom "LGHTAEOverrideLayoutChildren" "LGHTAEOverrideLayoutChildren_Replace" "childrenNumberRootOverride" "transfertNumChildrenRoot" "transfertNumChildrenRandRoot" "childProbabilityAlongBranchRoot";

			editorTemplate -callCustom "LGHTAERampControl" "LGHTAERampControl_Replace" "elevationAlongBranch";
			editorTemplate -callCustom "LGHTAERampControl" "LGHTAERampControl_Replace" "elevationRandAlongBranch";
			editorTemplate -callCustom "LGHTAEOverrideLayoutElevation" "LGHTAEOverrideLayoutElevation_Replace" "elevationRootOverride" "elevationAlongBranchRoot" "elevationRandAlongBranchRoot";

			editorTemplate -s "childrenNumber";
			editorTemplate -s "transfertNumChildren";
			editorTemplate -s "childrenNumberRand";
			editorTemplate -s "transfertNumChildrenRand";
			editorTemplate -s "childProbabilityAlongBranch";

			editorTemplate -s "childrenNumberRootOverride";
			editorTemplate -s "transfertNumChildrenRoot";
			editorTemplate -s "transfertNumChildrenRandRoot";			
			editorTemplate -s "elevationRootOverride";
			editorTemplate -s "elevationAlongBranchRoot";
			editorTemplate -s "elevationRandAlongBranchRoot";
		editorTemplate -endLayout;
		
		editorTemplate -s "inputCurve";


		AEdependNodeTemplate $nodeName;

	editorTemplate -addExtraControls;
	editorTemplate -endScrollLayout;
}

global proc LGHTAELineBaseTransfertAttribute ( string $name, string $att1S, string $att2S, int $isFloat )
{
	string $att1 = attName($att1S);
	string $att2 = attName($att2S);

	columnLayout -rowSpacing 2 -cat "left" 15 -adj true;
	rowLayout -nc 5 -cw5 130 35 55 100 30;
	if ($name == "")
	{
		$name = `attributeName -nice $att1S`;
	}
	text -label $name -align "left";
	text -label "base";
	if ($isFloat == 1)
	{
		floatField -w 35 ($att1+"_fldGrp");
	}
	else
	{
		intField -w 35 ($att1+"_fldGrp");	
	}
	text -label "transfert Factor" -align "right";
	floatField ($att2+"_fldGrp");
	setParent ..;
	setParent ..;

	LGHTAELineBaseTransfertLayout_Replace( $att1S, $att2S );
}

global proc LGHTAELineBaseIntTransfertLayout( string $baseAttS, string $transfertAttS )
{
	LGHTAELineBaseTransfertAttribute ( "", $baseAttS, $transfertAttS, 0);
}

global proc LGHTAELineBaseFloatTransfertLayout( string $baseAttS, string $transfertAttS )
{
	LGHTAELineBaseTransfertAttribute ( "", $baseAttS, $transfertAttS, 1);
}

global proc LGHTAELineBaseTransfertLayout_Replace( string $baseAttS, string $transfertAttS )
{
	string $base = attName($baseAttS);
	string $trans = attName($transfertAttS);
	connectControl ($base+"_fldGrp") $baseAttS;
	connectControl ($trans+"_fldGrp") $transfertAttS;	
}

global proc LGHTAELineBaseAttribute ( string $name, string $att1S, int $isFloat )
{
	string $att1 = attName($att1S);

	columnLayout -rowSpacing 2 -cat "left" 15 -adj true;
	rowLayout -nc 2 -cw2 165 55;
	if ($name == "")
	{
		$name = `attributeName -nice $att1S`;
	}
	text -label $name -align "left";
	if ($isFloat == 1)
	{
		floatField -w 35 ($att1+"_fldGrp");
	}
	else
	{
		intField -w 35 ($att1+"_fldGrp");	
	}
	setParent ..;
	setParent ..;

	LGHTAELineBaseLayout_Replace( $att1S );

}

global proc LGHTAELineBaseLayout_Replace( string $baseAttS )
{
	string $base = attName($baseAttS);
	connectControl ($base+"_fldGrp") $baseAttS;
}

global proc LGHTAELineBaseFloatAttribute ( string $att1S )
{
	LGHTAELineBaseAttribute ( "", $att1S, 1);
}

global proc LGHTAELineTransfertAttribute( string $name, string $attS )
{
	string $att = attName($attS);

	columnLayout -rowSpacing 2 -cat "left" 15 -adj true;
	rowLayout -nc 3 -cw3 145 100 30;
	if ($name == "")
		$name = `attributeName -nice $attS`;

	text -label $name -align "left";
	text -label "transfert Factor" -align "right";
	floatField ($att+"_fldGrp");
	setParent ..;
	setParent ..;
	LGHTAELineTransfertAttribute_Replace($attS);
}

global proc LGHTAELineTransfertAttribute_Replace( string $attS )
{
	string $att = attName($attS);
	connectControl ($att+"_fldGrp") $attS;	
}

global proc LGHTAETimeAccumulateLayout ( string $att1S, string $att2S )
{
	string $att1 = attName($att1S);
	string $att2 = attName($att2S);

	columnLayout -rowSpacing 2 -cat "left" 15 -adj true;
	rowLayout -nc 4 -cw4 130 70 140 60;
	$name1 = `attributeName -nice $att1S`;
	text -label $name1 -align "left";
	checkBox -label "" ($att1+"_chkBx");
	$name2 = `attributeName -nice $att2S`;
	text -label $name2 -align "left";
	floatField -w 50 ($att2+"_fldGrp");
	setParent ..;
	setParent ..;

	LGHTAETimeAccumulateLayout_Replace( $att1S, $att2S );
}

global proc LGHTAETimeAccumulateLayout_Replace ( string $att1S, string $att2S )
{
	string $att1 = attName($att1S);
	string $att2 = attName($att2S);
	connectControl ($att1+"_chkBx") $att1S;
	connectControl ($att2+"_fldGrp") $att2S;
}

global proc string attName( string $att )
{
	string $part[];
	tokenize($att, ".", $part);
	return $part[1];
}

global proc enableOverrideFL( string $fl, string $chb )
{
	int $v = `checkBox -q -v $chb`;
	frameLayout -e -vis ($v) $fl;
}

global proc LGHTAEOverrideLayoutGeneric( string $name, string $overrideAttS )
{
	string $overrideAtt = attName($overrideAttS);
	string $chkName = ($overrideAtt+"_CHB");
	string $flName = ($overrideAtt+"_FL");
	columnLayout -rowSpacing 1 -cat "left" 15 -adj true;
	checkBox -label ("Activate Root "+$name+" Overrides") $chkName;
	frameLayout -collapsable false -cl false -lv false -vis true -borderStyle "in" -mw 0 -mh 5 $flName;
	LGHTAEOverrideLayoutGeneric_Replace( $overrideAttS );
}

global proc LGHTAEOverrideLayoutGeneric_Replace ( string $overrideAttS )
{
	string $overrideAtt = attName($overrideAttS);
	connectControl ($overrideAtt+"_CHB") $overrideAttS;
}

global proc LGHTAEOverrideLayoutLength( string $overrideAttS , string $attRootTrS, string $attRootRpS )
{
	LGHTAEOverrideLayoutGeneric( "Length", $overrideAttS );
	LGHTAELineTransfertAttribute( "length", $attRootTrS );
	LGHTAERampControl( $attRootRpS );
}

global proc LGHTAEOverrideLayoutLength_Replace( string $overrideAttS , string $attRootTrS, string $attRootRpS )
{
	LGHTAEOverrideLayoutGeneric_Replace( $overrideAttS );
	LGHTAELineTransfertAttribute_Replace( $attRootTrS );
	LGHTAERampControl_Replace( $attRootRpS );
}

global proc LGHTAEOverrideLayoutIntensity( string $overrideAttS , string $attRoot1TrS, string $attRootRpS )
{
	LGHTAEOverrideLayoutGeneric( "Intensity", $overrideAttS );
	LGHTAELineTransfertAttribute( "intensity", $attRoot1TrS );
	LGHTAERampControl( $attRootRpS );
}

global proc LGHTAEOverrideLayoutIntensity_Replace( string $overrideAttS , string $attRoot1TrS, string $attRootRpS )
{
	LGHTAEOverrideLayoutGeneric_Replace( $overrideAttS );
	LGHTAELineTransfertAttribute_Replace( $attRoot1TrS );
	LGHTAERampControl_Replace( $attRootRpS );
}

global proc LGHTAEOverrideLayoutRadius( string $overrideAttS , string $attRoot1TrS, string $attRootRpS )
{
	LGHTAEOverrideLayoutGeneric( "Radius", $overrideAttS );
	LGHTAELineTransfertAttribute( "radius", $attRoot1TrS );
	LGHTAERampControl( $attRootRpS );
}

global proc LGHTAEOverrideLayoutRadius_Replace( string $overrideAttS , string $attRoot1TrS, string $attRootRpS )
{
	LGHTAEOverrideLayoutGeneric_Replace( $overrideAttS );
	LGHTAELineTransfertAttribute_Replace ($attRoot1TrS);
	LGHTAERampControl_Replace ($attRootRpS);
}

global proc LGHTAEOverrideLayoutTime( string $overrideAttS , string $attRoot1TrS, string $attRoot2TrS )
{
	LGHTAEOverrideLayoutGeneric( "Time", $overrideAttS );
	LGHTAELineTransfertAttribute( "Chaos Time Multiplier", $attRoot1TrS );
	LGHTAELineTransfertAttribute( "Skeleton Time Multiplier", $attRoot2TrS );
}

global proc LGHTAEOverrideLayoutTime_Replace( string $overrideAttS , string $attRoot1TrS, string $attRoot2TrS )
{
	LGHTAEOverrideLayoutGeneric_Replace( $overrideAttS );
	LGHTAELineTransfertAttribute_Replace ($attRoot1TrS);
	LGHTAELineTransfertAttribute_Replace ($attRoot2TrS);
}

global proc LGHTAEOverrideLayoutChildren( string $overrideAttS , string $attRoot1TrS, string $attRoot2TrS, string $attRootRpS )
{
	LGHTAEOverrideLayoutGeneric( "Children Nums", $overrideAttS );
	LGHTAELineTransfertAttribute( "Children Number", $attRoot1TrS );
	LGHTAELineTransfertAttribute( "Children Number Rand", $attRoot2TrS );
	LGHTAERampControl( $attRootRpS );
}

global proc LGHTAEOverrideLayoutChildren_Replace( string $overrideAttS , string $attRoot1TrS, string $attRoot2TrS, string $attRootRpS )
{
	LGHTAEOverrideLayoutGeneric_Replace( $overrideAttS );
	LGHTAELineTransfertAttribute_Replace ($attRoot1TrS);
	LGHTAELineTransfertAttribute_Replace ($attRoot2TrS);
	LGHTAERampControl_Replace( $attRootRpS );
}

global proc LGHTAEOverrideLayoutElevation( string $overrideAttS , string $attRoot1RpS, string $attRoot2RpS )
{
	LGHTAEOverrideLayoutGeneric( "Elevations", $overrideAttS );
	LGHTAERampControl( $attRoot1RpS );
	LGHTAERampControl( $attRoot2RpS );
}

global proc LGHTAEOverrideLayoutElevation_Replace( string $overrideAttS , string $attRoot1RpS, string $attRoot2RpS )
{
	LGHTAEOverrideLayoutGeneric_Replace( $overrideAttS );
	LGHTAERampControl_Replace( $attRoot1RpS );
	LGHTAERampControl_Replace( $attRoot2RpS );
}

global proc LGHTAEOverrideLayoutChaos( string $overrideAttS , string $attRoot1TrS, string $attRoot2TrS, string $attRoot1RpS )
{
	LGHTAEOverrideLayoutGeneric( "Chaos", $overrideAttS );
	LGHTAELineTransfertAttribute( "Chaos Displacement", $attRoot1TrS );
	LGHTAELineTransfertAttribute( "Chaos Frequency", $attRoot2TrS );
	LGHTAERampControl( $attRoot1RpS );
}

global proc LGHTAEOverrideLayoutChaos_Replace( string $overrideAttS , string $attRoot1TrS, string $attRoot2TrS, string $attRoot1RpS )
{
	LGHTAEOverrideLayoutGeneric_Replace( $overrideAttS );
	LGHTAELineTransfertAttribute_Replace( $attRoot1TrS );
	LGHTAELineTransfertAttribute_Replace( $attRoot2TrS );
	LGHTAERampControl_Replace( $attRoot1RpS );
}

			''')
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
			epsilon = 0.00001 # we clamp the values for childProbability, it's important since it remap array indices
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
			epsilon = 0.00001 # we clamp the values for childProbability, it's important since it remap array indices
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
			
			# Transfert of Along Branch dependent values
			transfertRadiusValue = acc.get( LBProcNode.transfertRadiusIndex)
			transfertChildLengthValue = acc.get( LBProcNode.transfertLengthIndex)
			transfertIntensityValue = acc.get( LBProcNode.transfertIntensityIndex)

			# Transfert of Generation dependent values
			transfertChaosOffsetValue = acc.get( LBProcNode.transfertChaosOffsetIndex)
			transfertSkeletonTimeValue = acc.get( LBProcNode.transfertSkeletonTimeIndex)
			transfertChaosTimeValue = acc.get( LBProcNode.transfertChaosTimeIndex)
			transfertNumChildrenValue = acc.get( LBProcNode.transfertNumChildrenIndex)
			transfertNumChildrenRandValue = acc.get( LBProcNode.transfertNumChildrenRandIndex)
			transfertChaosFrequencyValue = acc.get( LBProcNode.transfertChaosFrequencyIndex)
			transfertChaosVibrationValue = acc.get( LBProcNode.transfertChaosVibrationIndex)
			transfertLengthRandValue = acc.get( LBProcNode.transfertLengthRandIndex)

			# And the Root overrides that are not Ramp
			# Radius
			transfertRadiusRootValue = acc.getIf( LBProcNode.radiusRootOverrideIndex, LBProcNode.transfertRadiusRootIndex )
			# Childlength
			transfertChildLengthRootValue = acc.getIf( LBProcNode.lengthRootOverrideIndex, LBProcNode.transfertLengthRootIndex )
			# Intensity
			transfertIntensityRootValue = acc.getIf( LBProcNode.intensityRootOverrideIndex, LBProcNode.transfertIntensityRootIndex )
			# Times
			transfertChaosTimeRootValue = acc.getIf( LBProcNode.timeRootOverrideIndex, LBProcNode.transfertChaosTimeRootIndex )
			transfertSkeletonTimeRootValue = acc.getIf( LBProcNode.timeRootOverrideIndex, LBProcNode.transfertSkeletonTimeRootIndex )
			# Chaos
			transfertChaosOffsetRootValue = acc.getIf( LBProcNode.chaosDisplacementRootOverrideIndex, LBProcNode.transfertChaosOffsetRootIndex )
			transfertChaosFrequencyRootValue = acc.getIf( LBProcNode.chaosDisplacementRootOverrideIndex, LBProcNode.transfertChaosFrequencyRootIndex )
			# num Children
			transfertNumChildrenRootValue = acc.getIf( LBProcNode.childrenNumberRootOverrideIndex, LBProcNode.transfertNumChildrenRootIndex )
			transfertNumChildrenRandRootValue = acc.getIf( LBProcNode.childrenNumberRootOverrideIndex, LBProcNode.transfertNumChildrenRandRootIndex )

			tempCurves = acc.get(LBProcNode.inputCurvesIndex)
			if tempCurves == []:
				sys.stderr.write('lightninbolt has no curve to compute!\n')
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
			# the corresponding transfert values
			self.LP.setAPVTransfert( LBProcNode.LM.eAPBR.radius, transfertRadiusValue, transfertRadiusRootValue )
			self.LP.setAPVTransfert( LBProcNode.LM.eAPBR.intensity, transfertIntensityValue, transfertIntensityRootValue )
			self.LP.setAPVTransfert( LBProcNode.LM.eAPBR.length, transfertChildLengthValue, transfertChildLengthRootValue )

			# load the Generation inputs
			self.LP.setGENValue( LBProcNode.LM.eGEN.chaosFrequency, chaosFrequencyValue )
			self.LP.setGENValue( LBProcNode.LM.eGEN.chaosVibration, chaosVibrationValue )
			self.LP.setGENValue( LBProcNode.LM.eGEN.chaosTime, chaosTimeMultiplierValue )
			self.LP.setGENValue( LBProcNode.LM.eGEN.skeletonTime, skeletonTimeMultiplierValue )
			self.LP.setGENValue( LBProcNode.LM.eGEN.childrenNumber, childrenNumberValue )
			self.LP.setGENValue( LBProcNode.LM.eGEN.childrenNumberRand, childrenNumberRandValue )
			self.LP.setGENValue( LBProcNode.LM.eGEN.chaosDisplacement, chaosDisplacementMultValue )
			self.LP.setGENValue( LBProcNode.LM.eGEN.lengthRand, lengthRandValue )
			# the corresponding transfert values
			self.LP.setGENTransfert( LBProcNode.LM.eGEN.skeletonTime, transfertSkeletonTimeValue, transfertSkeletonTimeRootValue )
			self.LP.setGENTransfert( LBProcNode.LM.eGEN.chaosTime, transfertChaosTimeValue, transfertChaosTimeRootValue )
			self.LP.setGENTransfert( LBProcNode.LM.eGEN.chaosDisplacement, transfertChaosOffsetValue, transfertChaosOffsetRootValue )
			self.LP.setGENTransfert( LBProcNode.LM.eGEN.childrenNumber, transfertNumChildrenValue, transfertNumChildrenRootValue )
			self.LP.setGENTransfert( LBProcNode.LM.eGEN.childrenNumberRand, transfertNumChildrenRandValue, transfertNumChildrenRandRootValue )
			self.LP.setGENTransfert( LBProcNode.LM.eGEN.chaosFrequency, transfertChaosFrequencyValue, transfertChaosFrequencyRootValue )
			self.LP.setGENTransfert( LBProcNode.LM.eGEN.chaosVibration, transfertChaosVibrationValue )
			self.LP.setGENTransfert( LBProcNode.LM.eGEN.lengthRand, transfertLengthRandValue )
								
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

	# Generation Transfert Factors
	LBProcNode.MHLP.createAtt( 'transfertSkeletonTime', 'tst', eHlpT.double, eCG.main, default=3.0 )
	LBProcNode.MHLP.createAtt( 'transfertChaosTime', 'tct', eHlpT.double, eCG.main, default=1.0 )
	LBProcNode.MHLP.createAtt( 'transfertChaosFrequency', 'tcf', eHlpT.double, eCG.main, default=2.0 )
	LBProcNode.MHLP.createAtt( 'transfertChaosVibration', 'tcv', eHlpT.double, eCG.main, default=.6 )
	LBProcNode.MHLP.createAtt( 'transfertNumChildren', 'tnc', eHlpT.double, eCG.main, default=0.5 )
	LBProcNode.MHLP.createAtt( 'transfertNumChildrenRand', 'tncr', eHlpT.double, eCG.main, default=0.75 )
	LBProcNode.MHLP.createAtt( 'transfertChaosOffset', 'tco', eHlpT.double, eCG.main, default=0.6 )
	LBProcNode.MHLP.createAtt( 'transfertLengthRand', 'tlr', eHlpT.double, eCG.main, default=1.5 )

# APV Branches
	LBProcNode.MHLP.createAtt( 'radiusAlongBranch', 'rab', eHlpT.ramp, eCG.radiusAB )
	LBProcNode.MHLP.createAtt( 'radius', 'r', eHlpT.double, eCG.main, default=.15 )
	LBProcNode.MHLP.createAtt( 'childLengthAlongBranch', 'clab', eHlpT.ramp, eCG.childLengthAB )
	LBProcNode.MHLP.createAtt( 'length', 'l', eHlpT.double, eCG.main, default=1.0 )
	LBProcNode.MHLP.createAtt( 'intensityAlongBranch', 'iab', eHlpT.ramp, eCG.intensityAB )
	LBProcNode.MHLP.createAtt( 'intensity', 'i', eHlpT.double, eCG.main, default=1.0 )

	# APV Branches Transfert Factors
	LBProcNode.MHLP.createAtt( 'transfertRadius', 'tr', eHlpT.double, eCG.main, default=0.8 )
	LBProcNode.MHLP.createAtt( 'transfertLength', 'tl', eHlpT.double, eCG.main, default=0.6 )
	LBProcNode.MHLP.createAtt( 'transfertIntensity', 'ti', eHlpT.double, eCG.main, default=1.0 )

# APV Specials
	LBProcNode.MHLP.createAtt( 'chaosDisplacementAlongBranch', 'cdab', eHlpT.ramp, eCG.chaosDisplacementAB )

	# elevation 1 = 180 degree
	LBProcNode.MHLP.createAtt( 'elevationAlongBranch', 'eab', eHlpT.ramp, eCG.elevationAB )
	LBProcNode.MHLP.createAtt( 'elevationRandAlongBranch', 'erab', eHlpT.ramp, eCG.elevationRandAB )

	LBProcNode.MHLP.createAtt( 'childProbabilityAlongBranch', 'cpab', eHlpT.ramp, eCG.childProbabilityAB )

# Root Overrides
	# Generation Transfert Overrides
	LBProcNode.MHLP.createAtt( 'timeRootOverride', 'tro', eHlpT.bool, eCG.main, default=False )
	LBProcNode.MHLP.createAtt( 'transfertChaosTimeRoot', 'tctrt', eHlpT.double, eCG.main, default=1.0 )
	LBProcNode.MHLP.createAtt( 'transfertSkeletonTimeRoot', 'tstrt', eHlpT.double, eCG.main, default=1.0 )

	LBProcNode.MHLP.createAtt( 'chaosDisplacementRootOverride', 'coro', eHlpT.bool, eCG.chaosDisplacementABRoot, default=True )
	LBProcNode.MHLP.createAtt( 'chaosDisplacementAlongBranchRoot', 'orrt', eHlpT.ramp, eCG.chaosDisplacementABRoot )
	LBProcNode.MHLP.createAtt( 'transfertChaosOffsetRoot', 'tor', eHlpT.double, eCG.main, default=0.55 )
	LBProcNode.MHLP.createAtt( 'transfertChaosFrequencyRoot', 'tcfr', eHlpT.double, eCG.main, default=2.0 )

	LBProcNode.MHLP.createAtt( 'childrenNumberRootOverride', 'ncro', eHlpT.bool, eCG.main, default=0 )
	LBProcNode.MHLP.createAtt( 'transfertNumChildrenRoot', 'tncrt', eHlpT.double, eCG.main, default=1.0 )
	LBProcNode.MHLP.createAtt( 'transfertNumChildrenRandRoot', 'tncrrt', eHlpT.double, eCG.main, default=1.0 )
	LBProcNode.MHLP.createAtt( 'childProbabilityAlongBranchRoot', 'cpabr', eHlpT.ramp, eCG.childProbabilityABRoot )


	# APV Special overrides
	LBProcNode.MHLP.createAtt( 'elevationRootOverride', 'ero', eHlpT.bool, [eCG.elevationABRoot, eCG.elevationRandABRoot], default=False )
	LBProcNode.MHLP.createAtt( 'elevationAlongBranchRoot', 'errt', eHlpT.ramp, eCG.elevationABRoot )	
	LBProcNode.MHLP.createAtt( 'elevationRandAlongBranchRoot', 'errrt', eHlpT.ramp, eCG.elevationRandABRoot )

	# APV Branches overrides
	LBProcNode.MHLP.createAtt( 'radiusRootOverride', 'rro', eHlpT.bool, eCG.radiusABRoot, default=True )
	LBProcNode.MHLP.createAtt( 'radiusAlongBranchRoot', 'rrrt', eHlpT.ramp, eCG.radiusABRoot )
	LBProcNode.MHLP.createAtt( 'transfertRadiusRoot', 'trr', eHlpT.double, eCG.main, default=0.45 )
	
	LBProcNode.MHLP.createAtt( 'lengthRootOverride', 'lro', eHlpT.bool, eCG.childLengthABRoot, default=False )
	LBProcNode.MHLP.createAtt( 'childLengthAlongBranchRoot', 'clapr', eHlpT.ramp, eCG.childLengthABRoot )
	LBProcNode.MHLP.createAtt( 'transfertLengthRoot', 'tlro', eHlpT.double, eCG.main, default=1.0 )
	
	LBProcNode.MHLP.createAtt( 'intensityRootOverride', 'iro', eHlpT.bool, eCG.intensityABRoot, default=True )
	LBProcNode.MHLP.createAtt( 'intensityAlongBranchRoot', 'iabr', eHlpT.ramp, eCG.intensityABRoot )
	LBProcNode.MHLP.createAtt( 'transfertIntensityRoot', 'tir', eHlpT.double, eCG.main, default=1.0 )

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
