#!/usr/bin/python
# -*- coding: iso-8859-1 -*-


'''

import pymel.core as pm
pm.loadPlugin('/Users/mathieu/Documents/IMPORTANT/Perso/Prog/python/lightningProcessor/mayaLightningPlugin.py')

lightNode = pm.createNode('lightningbolt')
meshNode = pm.createNode('mesh')
pm.connectAttr( 'time1.outTime', lightNode+'.time')
pm.connectAttr( 'curveShape1.worldSpace[0]', lightNode+'.inputCurve')
pm.connectAttr( lightNode+'.outputMesh', meshNode+'.inMesh')

mrVert = pm.createNode('mentalrayVertexColors')
sh = pm.shadingNode( 'surfaceShader', asShader=True )
set = pm.sets( renderable=True, noSurfaceShader=True, empty=True)
pm.connectAttr( sh+'.outColor', set+'.surfaceShader', f=True)
pm.sets( set, e=True, forceElement=meshNode )
forceEval = pm.polyEvaluate( meshNode, v=True )
pm.connectAttr( (meshNode+".colorSet[0].colorName") , (mrVert+".cpvSets[0]") )
pm.connectAttr( (mrVert+'.outColor'), (sh+".outColor") )
'''

import sys
import maya.OpenMaya as OpenMaya
import maya.OpenMayaMPx as OpenMayaMPx
import maya.mel as mel


kPluginNodeName = "lightningbolt"
kPluginNodeId = OpenMaya.MTypeId(0x8700B)

def enum(*sequential, **named):
	enums = dict(zip(sequential, range(len(sequential))), **named)
	return type('Enum', (), enums)

#------------------- Here is the Helper Class to make to process of creating, accessing and setting the influence of attributes less tedious
class MAH_msCommandException(Exception):
    def __init__(self,message):
        self.message = '[MAH] '+message
    
    def __str__(self):
        return self.message

class attDef:
	def __init__( self, index, mObject, name , shortName, type,  addAttr, isInput  ):
		self.mObject = mObject
		self.index = index
		self.name = name
		self.shortName = shortName
		self.type = type
		self.addAttr = addAttr
		self.isInput = isInput

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
				#sys.stderr.write("plug "+str(plug.name())+" is called for compute\n")
				needCompute = True
				break
		if needCompute:
			for dep in self.dependencies :
				dep.forceEvaluateOUTS(thisNode)
		return needCompute

	def setClean( self, data ):
		for attOut in self.OUTS:
			data.setClean( attOut.mObject )
		#for attOut in self.OUTS:
			#sys.stderr.write(" plug out "+attOut.name+" clean "+str(data.isClean( attOut.mObject ))+"\n")

eHlpT = enum('double', 'int', 'bool', 'time', 'curve', 'mesh', 'ramp', 'maxType') 
aHlpMayaTypes = [ OpenMaya.MFnNumericData.kDouble, OpenMaya.MFnNumericData.kInt, OpenMaya.MFnNumericData.kBoolean, OpenMaya.MFnUnitAttribute.kTime, OpenMaya.MFnData.kNurbsCurve, OpenMaya.MFnData.kMesh, '' ]

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

	def createAtt( self, name, shortName, type, computeGpIds, isInput=True, default=None, addAttr=True ):
		attIndex = len(self.attributes)
		sys.stderr.write('adding attribute '+name+'\n')

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

		att = attDef(attIndex, mObject, name, shortName, type,  addAttr, isInput )
		
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
		#globalInList = set().union( globalInList , computeGp.OUTS )
		return globalInList

	def finalizeAttributeInitialization( self ):
		#---- first check the compute groups, if one has no OUT then create one "dummy" out for it
		sys.stderr.write('adding dummies \n')
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
		sys.stderr.write('adding atts \n')
		for a in self.attributes:
			if not a.addAttr: # some attributes don't need to be added like double3
				continue
			self.nodeClass.addAttribute( a.mObject )
		#---- then we can declare affects
		sys.stderr.write('declare affects \n')
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

	def getAttValueOrHdl( self, attId, thisNode, data ):
		attDef = self.attributes[attId]

		if not attDef.isInput:
			return data.outputValue( attDef.mObject )

		if attDef.type == eHlpT.ramp :
			RampPlug = OpenMaya.MPlug( thisNode, attDef.mObject )
			RampPlug.setAttribute(attDef.mObject)
			RampPlug01 = OpenMaya.MPlug(RampPlug.node(), RampPlug.attribute())
			return OpenMaya.MRampAttribute(RampPlug01.node(), RampPlug01.attribute())
		
		tempData = data.inputValue(attDef.mObject)

		if attDef.type == eHlpT.double :
			return tempData.asDouble()
		if attDef.type == eHlpT.bool :
			return tempData.asBool()
		if attDef.type == eHlpT.time :
			return tempData.asTime().value()
		if attDef.type == eHlpT.int :
			return tempData.asInt()
		if attDef.type == eHlpT.curve :
			return tempData.asNurbsCurveTransformed()
		#if attDef.type is "vectorOfDouble" :
		#	return OpenMaya.MVector( tempData.asVector() )
		raise MAH_msCommandException('type not handled in getAttValueOrHdl!')

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

#######################################################

import numpy as np
import math

class MLG_msCommandException(Exception):
    def __init__(self,message):
        self.message = '[MLG] '+message
    
    def __str__(self):
        return self.message

# Compute groups used by lightning Node
eCG = enum( 'main', 'detail',
			'radiusAB', 'childLengthAB', 'intensityAB', 'chaosOffsetAB', 'elevationAB', 'elevationRandAB', 'childProbabilityAB',
			'chaosOffsetABRoot', 'elevationABRoot', 'elevationRandABRoot', 'radiusABRoot', 'childLengthABRoot', 'intensityABRoot',
			'max' )

# Function used to convert the Processor output into a Maya mesh structure
def mayaLightningMesher(resultLoopingPoints, numLoopingLightningBranch, resultPoints, numLightningBranch, resultIntensityColors, branchMaxPoints, tubeSide):
	
	facesConnect = [ ((id/2+1)%2 * ( ( id%4 + id/4 )%tubeSide) + (id/2)%2 *   (( (( 3-id%4 + id/4 )  )%tubeSide) + tubeSide)) + ring*tubeSide + branchNum*tubeSide*branchMaxPoints for branchNum in range(numLightningBranch) for ring in range(branchMaxPoints-1) for id in range(tubeSide*4) ]
	
	facesCount = [4]*(tubeSide*(branchMaxPoints-1)*numLightningBranch)

	numVertices = len(resultPoints)/4
	numFaces = len(facesCount)
	#sys.stderr.write("vertices "+str(numVertices)+" \n")
	#sys.stderr.write("faces "+str(numFaces)+" \n")

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
	
	#sys.stderr.write("mesh contruit \n")
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

def fillArrayFromRampAtt( array, ramp):
	inc = 1.0/(array.size-1)
	param = 0.0
	valAt_util = OpenMaya.MScriptUtil()
	valAt_util.createFromDouble(1.0)
	valAtPtr = valAt_util.asFloatPtr()
	for i in range(array.size):
		ramp.getValueAtPosition(param,valAtPtr)
		array[i] = valAt_util.getFloat(valAtPtr)
		param=param+inc	

def setupABVFromRamp( arrays, ramp):
	array, arrayRoot = arrays
	if array.size <1 :
		raise MLG_msCommandException('array size must be more than 1')
	fillArrayFromRampAtt( array, ramp)
	sys.stderr.write('array '+str(array)+'\n')

def setupABRootVFromRamp( arrays, ramp):
	array, arrayRoot = arrays
	if arrayRoot.size <1 :
		raise MLG_msCommandException('array size must be more than 1')
	if ramp is None :
		arrayRoot[:] = array # COPY Normal -> Root
		return		
	fillArrayFromRampAtt( arrayRoot, ramp)


def getPointListFromCurve( numPoints, fnCurve, lengthScale):
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
	
	step = lengthScale*(paramEnd - paramStart)/(numPoints-1)
	
	pt=OpenMaya.MPoint()
	param=paramStart
	res = []
	#fullLength = 0
	for i in range(numPoints):
		#sys.stderr.write('i '+str(i)+' param '+str(param)+'\n')
		fnCurve.getPointAtParam(param, pt)
		res.append( [pt.x,pt.y,pt.z] )
		#if i>0 :
		#	fullLength += math.sqrt(res[i-1][0]*pt.x + res[i-1][1]*pt.y + res[i-1][2]*pt.z) 
		#sys.stderr.write('fullLength '+str(fullLength)+'\n')

		param += step
	return res, lengthScale*fnCurve.length()


class lightningBoltNode(OpenMayaMPx.MPxNode):
	#----------------------------------------------------------------------------
	##AEtemplate proc
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

global proc AElightningboltTemplate( string $nodeName )
{
	AEswatchDisplay  $nodeName;
	editorTemplate -beginScrollLayout;

		editorTemplate -beginLayout "System parameters" -collapse 1;
			editorTemplate -addControl "time";
			editorTemplate -addControl "tubeSides";
			editorTemplate -addControl "maxGeneration";
			editorTemplate -addControl "detail";
			editorTemplate -addControl "seedChaos";
			editorTemplate -addControl "seedBranching";
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
			editorTemplate -callCustom "LGHTAERampControl" "LGHTAERampControl_Replace" "intensityAlongPath";
			editorTemplate -callCustom "LGHTAEOverrideLayoutIntensity" "LGHTAEOverrideLayoutIntensity_Replace" "intensityRootOverride" "transfertIntensityRoot" "intensityAlongPathRoot";

			editorTemplate -s "intensity";
			editorTemplate -s "transfertIntensity";
			editorTemplate -s "intensityAlongPath";
			editorTemplate -s "intensityRootOverride";
			editorTemplate -s "intensityAlongPathRoot";
			editorTemplate -s "transfertIntensityRoot";	
		editorTemplate -endLayout;

		editorTemplate -beginLayout "⟦ Time Attributes ⟧" -collapse 1;
			editorTemplate -callCustom "LGHTAELineBaseFloatTransfertLayout" "LGHTAELineBaseTransfertLayout_Replace" "timeChaos" "transfertTimeChaos";
			editorTemplate -addControl "vibrationTimeFactor";
			editorTemplate -callCustom "LGHTAELineBaseFloatTransfertLayout" "LGHTAELineBaseTransfertLayout_Replace" "timeBranching" "transfertTimeBranching";
			editorTemplate -callCustom "LGHTAEOverrideLayoutTime" "LGHTAEOverrideLayoutTime_Replace" "timeRootOverride" "transfertTimeChaosRoot" "transfertTimeBranchingRoot";				
			editorTemplate -s "timeChaos";
			editorTemplate -s "vibrationTimeFactor";
			editorTemplate -s "transfertTimeChaos";
			editorTemplate -s "timeBranching";
			editorTemplate -s "transfertTimeBranching";
			editorTemplate -s "timeRootOverride";
			editorTemplate -s "transfertTimeChaosRoot";
			editorTemplate -s "transfertTimeBranchingRoot";	
		editorTemplate -endLayout;
	
		editorTemplate -beginLayout "⟦ Chaos Attributes ⟧" -collapse 1;
			editorTemplate -callCustom "LGHTAELineBaseFloatTransfertLayout" "LGHTAELineBaseTransfertLayout_Replace" "chaosOffset" "transfertChaosOffset";
			editorTemplate -callCustom "LGHTAELineBaseFloatTransfertLayout" "LGHTAELineBaseTransfertLayout_Replace" "chaosFrequency" "transfertChaosFrequency";
			editorTemplate -addControl "secondaryChaosFreqFactor";
			editorTemplate -addControl "secondaryChaosMinClamp";
			editorTemplate -addControl "secondaryChaosMaxClamp";
			editorTemplate -addControl "secondaryChaosMinRemap";
			editorTemplate -addControl "secondaryChaosMaxRemap";

			editorTemplate -callCustom "LGHTAELineBaseFloatTransfertLayout" "LGHTAELineBaseTransfertLayout_Replace" "chaosVibration" "transfertChaosVibration";
			editorTemplate -callCustom "LGHTAERampControl" "LGHTAERampControl_Replace" "chaosOffsetAlongBranch";
			editorTemplate -callCustom "LGHTAEOverrideLayoutChaos" "LGHTAEOverrideLayoutChaos_Replace" "chaosOffsetRootOverride" "transfertChaosOffsetRoot" "transfertChaosFrequencyRoot" "chaosOffsetAlongBranchRoot";			

			editorTemplate -s "chaosOffsetRootOverride";
			editorTemplate -s "transfertChaosOffsetRoot";
			editorTemplate -s "chaosOffsetAlongBranchRoot";
			editorTemplate -s "transfertChaosFrequencyRoot";

			editorTemplate -s "chaosOffset";
			editorTemplate -s "transfertChaosOffset";
			editorTemplate -s "chaosFrequency";
			editorTemplate -s "transfertChaosFrequency";
			editorTemplate -s "chaosVibration";
			editorTemplate -s "transfertChaosVibration";

			editorTemplate -s "chaosOffsetAlongBranch";
		editorTemplate -endLayout;

		editorTemplate -beginLayout "⟦ Childs Generation Attributes ⟧" -collapse 1;
			editorTemplate -callCustom "LGHTAELineBaseIntTransfertLayout" "LGHTAELineBaseTransfertLayout_Replace" "numChildren" "transfertNumChildren";
			editorTemplate -callCustom "LGHTAELineBaseIntTransfertLayout" "LGHTAELineBaseTransfertLayout_Replace" "numChildrenRand" "transfertNumChildrenRand";
			editorTemplate -callCustom "LGHTAERampControl" "LGHTAERampControl_Replace" "childProbabilityAlongBranch";
			editorTemplate -callCustom "LGHTAEOverrideLayoutChildren" "LGHTAEOverrideLayoutChildren_Replace" "numChildrenRootOverride" "transfertNumChildrenRoot" "transfertNumChildrenRandRoot";

			editorTemplate -callCustom "LGHTAERampControl" "LGHTAERampControl_Replace" "elevationAlongBranch";
			editorTemplate -callCustom "LGHTAERampControl" "LGHTAERampControl_Replace" "elevationRandAlongBranch";
			editorTemplate -callCustom "LGHTAEOverrideLayoutElevation" "LGHTAEOverrideLayoutElevation_Replace" "elevationRootOverride" "elevationAlongBranchRoot" "elevationRandAlongBranchRoot";

			editorTemplate -s "numChildren";
			editorTemplate -s "transfertNumChildren";
			editorTemplate -s "numChildrenRand";
			editorTemplate -s "transfertNumChildrenRand";
			editorTemplate -s "childProbabilityAlongBranch";

			editorTemplate -s "numChildrenRootOverride";
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
	rowLayout -nc 5 -cw5 110 35 55 100 30;
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

global proc LGHTAELineTransfertAttribute( string $name, string $attS )
{
	string $att = attName($attS);

	columnLayout -rowSpacing 2 -cat "left" 15 -adj true;
	rowLayout -nc 3 -cw3 110 100 30;
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

global proc string attName( string $att )
{
	string $part[];
	tokenize($att, ".", $part);
	return $part[1];
}

global proc enableOverrideFL( string $fl, string $chb )
{
	int $v = `checkBox -q -v $chb`;
	//frameLayout -e -cl (!$v) $fl;
	frameLayout -e -vis ($v) $fl;
}

global proc LGHTAEOverrideLayoutGeneric( string $name, string $overrideAttS )
{
	string $overrideAtt = attName($overrideAttS);
	string $chkName = ($overrideAtt+"_CHB");
	string $flName = ($overrideAtt+"_FL");
	columnLayout -rowSpacing 1 -cat "left" 15 -adj true;
	checkBox -label ("Activate Root "+$name+" Overrides") $chkName;
	//checkBox -label ("Root "+$name+" Overrides") -cc ( "enableOverrideFL( \\""+$flName+"\\" ,\\""+$chkName+"\\")" ) $chkName;
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
	LGHTAELineTransfertAttribute( "timeChaos", $attRoot1TrS );
	LGHTAELineTransfertAttribute( "timeBranching", $attRoot2TrS );
}

global proc LGHTAEOverrideLayoutTime_Replace( string $overrideAttS , string $attRoot1TrS, string $attRoot2TrS )
{
	LGHTAEOverrideLayoutGeneric_Replace( $overrideAttS );
	LGHTAELineTransfertAttribute_Replace ($attRoot1TrS);
	LGHTAELineTransfertAttribute_Replace ($attRoot2TrS);
}

global proc LGHTAEOverrideLayoutChildren( string $overrideAttS , string $attRoot1TrS, string $attRoot2TrS )
{
	LGHTAEOverrideLayoutGeneric( "Children Nums", $overrideAttS );
	LGHTAELineTransfertAttribute( "numChildren", $attRoot1TrS );
	LGHTAELineTransfertAttribute( "numChildrenRand", $attRoot2TrS );
}

global proc LGHTAEOverrideLayoutChildren_Replace( string $overrideAttS , string $attRoot1TrS, string $attRoot2TrS )
{
	LGHTAEOverrideLayoutGeneric_Replace( $overrideAttS );
	LGHTAELineTransfertAttribute_Replace ($attRoot1TrS);
	LGHTAELineTransfertAttribute_Replace ($attRoot2TrS);
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
	LGHTAELineTransfertAttribute( "chaosOffset", $attRoot1TrS );
	LGHTAELineTransfertAttribute( "chaosFrequency", $attRoot2TrS );
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
	nhlp = None
	#----------------------------------------------------------------------------

	def __init__(self):
		OpenMayaMPx.MPxNode.__init__(self)
		self.LP = lightningBoltNode.LM.lightningBoltProcessor(mayaLightningMesher) # instance a processor for this node

	def compute(self, plug, data):
		thisNode = self.thisMObject()
		acc = ATTAccessor( lightningBoltNode.nhlp, thisNode, data )
	# -- Compute Detail
		if acc.needCompute( eCG.detail, plug ):
			detailValue = acc.get(lightningBoltNode.detailIndex)
			self.LP.setGlobalValue( lightningBoltNode.LM.eGLOBAL.detail, detailValue ) # ***** initialisation of the sizes of arrays in the Processor (all Along Path arrays depend on detail)
			acc.setClean(eCG.detail, data)
	# -- Compute Along Branch Ramp : Radius
		elif acc.needCompute( eCG.radiusAB, plug ):
			RadiusABRamp = acc.get(lightningBoltNode.radiusAlongBranchIndex)	
			setupABVFromRamp( self.LP.getAPBR( lightningBoltNode.LM.eAPBR.radius ), RadiusABRamp )
			if not acc.get( lightningBoltNode.radiusRootOverrideIndex ): # if no Root override then copy
				setupABRootVFromRamp( self.LP.getAPBR( lightningBoltNode.LM.eAPBR.radius ), None )
			acc.setClean(eCG.radiusAB, data)
	# -- Compute Along Branch Ramp : Intensity
		elif acc.needCompute( eCG.intensityAB, plug ):
			IntensityABRamp = acc.get(lightningBoltNode.intensityAlongPathIndex)	
			setupABVFromRamp( self.LP.getAPBR( lightningBoltNode.LM.eAPBR.intensity ), IntensityABRamp )
			if not acc.get( lightningBoltNode.intensityRootOverrideIndex ): # if no Root override then copy
				setupABRootVFromRamp( self.LP.getAPBR( lightningBoltNode.LM.eAPBR.intensity ), None )
			acc.setClean(eCG.intensityAB, data)
	# -- Compute Along Branch Ramp : Child Length
		elif acc.needCompute( eCG.childLengthAB, plug ):
			childLengthABRamp = acc.get(lightningBoltNode.childLengthAlongBranchIndex)	
			setupABVFromRamp( self.LP.getAPBR( lightningBoltNode.LM.eAPBR.length ), childLengthABRamp )
			if not acc.get( lightningBoltNode.lengthRootOverrideIndex ): # if no Root override then copy
				setupABRootVFromRamp( self.LP.getAPBR( lightningBoltNode.LM.eAPBR.length ), None )
			acc.setClean(eCG.childLengthAB, data)
	# -- Compute Along Branch Ramp : Chaos Offset
		elif acc.needCompute( eCG.chaosOffsetAB, plug ):
			chaosOffsetABRamp = acc.get(lightningBoltNode.chaosOffsetAlongBranchIndex)	
			setupABVFromRamp( self.LP.getAPSPE( lightningBoltNode.LM.eAPSPE.chaosOffset ), chaosOffsetABRamp )
			if not acc.get( lightningBoltNode.chaosOffsetRootOverrideIndex ): # if no Root override then copy
				setupABRootVFromRamp( self.LP.getAPSPE( lightningBoltNode.LM.eAPSPE.chaosOffset ), None )
			acc.setClean(eCG.chaosOffsetAB, data)
	# -- Compute Along Branch Ramp : Elevation
		elif acc.needCompute( eCG.elevationAB, plug ):
			elevationABRamp = acc.get(lightningBoltNode.elevationAlongBranchIndex)
			setupABVFromRamp( self.LP.getAPSPE( lightningBoltNode.LM.eAPSPE.elevation ), elevationABRamp )
			if not acc.get( lightningBoltNode.elevationRootOverrideIndex ): # if no Root override then copy
				setupABRootVFromRamp( self.LP.getAPSPE( lightningBoltNode.LM.eAPSPE.elevation ), None )
			acc.setClean(eCG.elevationAB, data)
	# -- Compute Along Branch Ramp : Elevation Rand
		elif acc.needCompute( eCG.elevationRandAB, plug ):
			elevationRandABRamp = acc.get(lightningBoltNode.elevationRandAlongBranchIndex)	
			setupABVFromRamp( self.LP.getAPSPE( lightningBoltNode.LM.eAPSPE.elevationRand ), elevationRandABRamp )
			if not acc.get( lightningBoltNode.elevationRootOverrideIndex ): # if no Root override then copy
				setupABRootVFromRamp( self.LP.getAPSPE( lightningBoltNode.LM.eAPSPE.elevationRand ), None )
			acc.setClean(eCG.elevationRandAB, data)
	# -- Compute Along Branch Ramp : Child Probability
		elif acc.needCompute( eCG.childProbabilityAB, plug ):
			childProbabilityABRamp = acc.get(lightningBoltNode.childProbabilityAlongBranchIndex)
			setupABVFromRamp( self.LP.getAPSPE( lightningBoltNode.LM.eAPSPE.childProbability ), childProbabilityABRamp )
			# child probability need a copy of the normal values to the root because there is no override to do it
			setupABRootVFromRamp( self.LP.getAPSPE( lightningBoltNode.LM.eAPSPE.childProbability ), None )
			acc.setClean(eCG.childProbabilityAB, data)
	# -- Compute ROOT Along Branch Ramp : Radius
		elif acc.needCompute( eCG.radiusABRoot, plug ):
			radiusABRootRamp = acc.get(lightningBoltNode.radiusAlongBranchRootIndex)	
			if not acc.get( lightningBoltNode.radiusRootOverrideIndex ):
				radiusABRootRamp = None # if none the values of the Normal Ramp are copied to the Root Ramp
			setupABRootVFromRamp( self.LP.getAPBR( lightningBoltNode.LM.eAPBR.radius ), radiusABRootRamp )
			acc.setClean(eCG.radiusABRoot, data)
	# -- Compute ROOT Along Branch Ramp : Child Length
		elif acc.needCompute( eCG.childLengthABRoot, plug ):
			childLengthABRootRamp = acc.get(lightningBoltNode.childLengthAlongBranchRootIndex)	
			if not acc.get( lightningBoltNode.lengthRootOverrideIndex ):
				childLengthABRootRamp = None # if none the values of the Normal Ramp are copied to the Root Ramp
			setupABRootVFromRamp( self.LP.getAPBR( lightningBoltNode.LM.eAPBR.length ), childLengthABRootRamp )
			acc.setClean(eCG.childLengthABRoot, data)
	# -- Compute ROOT Along Branch Ramp : Intensity
		elif acc.needCompute( eCG.intensityABRoot, plug ):
			intensityABRootRamp = acc.get(lightningBoltNode.intensityAlongPathRootIndex)	
			if not acc.get( lightningBoltNode.intensityRootOverrideIndex ):
				intensityABRootRamp = None # if none the values of the Normal Ramp are copied to the Root Ramp
			setupABRootVFromRamp( self.LP.getAPBR( lightningBoltNode.LM.eAPBR.intensity ), intensityABRootRamp )
			acc.setClean(eCG.intensityABRoot, data)
	# -- Compute ROOT Along Branch Ramp : Chaos Offset
		elif acc.needCompute( eCG.chaosOffsetABRoot, plug ):
			chaosOffsetABRootRamp = acc.get(lightningBoltNode.chaosOffsetAlongBranchRootIndex)	
			if not acc.get( lightningBoltNode.chaosOffsetRootOverrideIndex ):
				chaosOffsetABRootRamp = None # if none the values of the Normal Ramp are copied to the Root Ramp
			setupABRootVFromRamp( self.LP.getAPSPE( lightningBoltNode.LM.eAPSPE.chaosOffset ), chaosOffsetABRootRamp )
			acc.setClean(eCG.chaosOffsetABRoot, data)
	# -- Compute ROOT Along Branch Ramp : Elevation
		elif acc.needCompute( eCG.elevationABRoot, plug ):
			elevationABRootRamp = acc.get(lightningBoltNode.elevationAlongBranchRootIndex)	
			if not acc.get( lightningBoltNode.elevationRootOverrideIndex ):
				elevationABRootRamp = None
			#sys.stderr.write('elevation ramp Root '+str(elevationABRootRamp)+' \n')
			setupABRootVFromRamp( self.LP.getAPSPE( lightningBoltNode.LM.eAPSPE.elevation ), elevationABRootRamp )
			acc.setClean(eCG.elevationABRoot, data)
	# -- Compute ROOT Along Branch Ramp : Elevation Rand
		elif acc.needCompute( eCG.elevationRandABRoot, plug ):
			elevationRandABRootRamp = acc.get(lightningBoltNode.elevationRandAlongBranchRootIndex)	
			if not acc.get( lightningBoltNode.elevationRootOverrideIndex ):
				elevationRandABRootRamp = None
			setupABRootVFromRamp( self.LP.getAPSPE( lightningBoltNode.LM.eAPSPE.elevationRand ), elevationRandABRootRamp )
			acc.setClean(eCG.elevationRandABRoot, data)
	# -- Compute MAIN
		elif acc.needCompute( eCG.main, plug ):
		#-------- Read All the attributes from plugs
			# global values
			timeValue = acc.get( lightningBoltNode.timeIndex)
			tubeSidesValue = acc.get( lightningBoltNode.tubeSidesIndex)
			maxGenerationValue = acc.get( lightningBoltNode.maxGenerationIndex)
			vibrationTimeFactorValue = acc.get( lightningBoltNode.vibrationTimeFactorIndex)
			secondaryChaosFreqFactorValue = acc.get( lightningBoltNode.secondaryChaosFreqFactorIndex )
			secondaryChaosMinClampValue = acc.get( lightningBoltNode.secondaryChaosMinClampIndex )
			secondaryChaosMaxClampValue = acc.get( lightningBoltNode.secondaryChaosMaxClampIndex )
			secondaryChaosMinRemapValue = acc.get( lightningBoltNode.secondaryChaosMinRemapIndex )
			secondaryChaosMaxRemapValue = acc.get( lightningBoltNode.secondaryChaosMaxRemapIndex )
			seedChaosValue = acc.get( lightningBoltNode.seedChaosIndex)
			seedBranchingValue = acc.get( lightningBoltNode.seedBranchingIndex)

			# Along Branch dependent values
			radiusValue = acc.get( lightningBoltNode.radiusIndex)
			intensityValue = acc.get( lightningBoltNode.intensityIndex)
			childLengthValue = acc.get( lightningBoltNode.lengthIndex)

			# Generation dependent values
			chaosFrequencyValue = acc.get( lightningBoltNode.chaosFrequencyIndex)
			chaosVibrationValue = acc.get( lightningBoltNode.chaosVibrationIndex)
			numChildrenValue = acc.get( lightningBoltNode.numChildrenIndex)
			numChildrenRandValue = acc.get( lightningBoltNode.numChildrenRandIndex)
			timeChaosValue = acc.get( lightningBoltNode.timeChaosIndex)
			timeBranchingValue = acc.get( lightningBoltNode.timeBranchingIndex)
			chaosOffsetMultValue = acc.get( lightningBoltNode.chaosOffsetIndex)
			lengthRandValue = acc.get( lightningBoltNode.lengthRandIndex)
			
			# Transfert of Along Branch dependent values
			transfertRadiusValue = acc.get( lightningBoltNode.transfertRadiusIndex)
			transfertChildLengthValue = acc.get( lightningBoltNode.transfertLengthIndex)
			transfertIntensityValue = acc.get( lightningBoltNode.transfertIntensityIndex)

			# Transfert of Generation dependent values
			transfertChaosOffsetValue = acc.get( lightningBoltNode.transfertChaosOffsetIndex)
			transfertTimeBranchingValue = acc.get( lightningBoltNode.transfertTimeBranchingIndex)
			transfertTimeChaosValue = acc.get( lightningBoltNode.transfertTimeChaosIndex)
			transfertNumChildrenValue = acc.get( lightningBoltNode.transfertNumChildrenIndex)
			transfertNumChildrenRandValue = acc.get( lightningBoltNode.transfertNumChildrenRandIndex)
			transfertChaosFrequencyValue = acc.get( lightningBoltNode.transfertChaosFrequencyIndex)
			transfertChaosVibrationValue = acc.get( lightningBoltNode.transfertChaosVibrationIndex)
			transfertLengthRandValue = acc.get( lightningBoltNode.transfertLengthRandIndex)

			# And the Root overrides that are not Ramp
			# Radius
			transfertRadiusRootValue = acc.getIf( lightningBoltNode.radiusRootOverrideIndex, lightningBoltNode.transfertRadiusRootIndex )
			# Childlength
			transfertChildLengthRootValue = acc.getIf( lightningBoltNode.lengthRootOverrideIndex, lightningBoltNode.transfertLengthRootIndex )
			# Intensity
			transfertIntensityRootValue = acc.getIf( lightningBoltNode.intensityRootOverrideIndex, lightningBoltNode.transfertIntensityRootIndex )
			# Times
			transfertTimeChaosRootValue = acc.getIf( lightningBoltNode.timeRootOverrideIndex, lightningBoltNode.transfertTimeChaosRootIndex )
			transfertTimeBranchingRootValue = acc.getIf( lightningBoltNode.timeRootOverrideIndex, lightningBoltNode.transfertTimeBranchingRootIndex )
			# Chaos
			transfertChaosOffsetRootValue = acc.getIf( lightningBoltNode.chaosOffsetRootOverrideIndex, lightningBoltNode.transfertChaosOffsetRootIndex )
			transfertChaosFrequencyRootValue = acc.getIf( lightningBoltNode.chaosOffsetRootOverrideIndex, lightningBoltNode.transfertChaosFrequencyRootIndex )
			# num Children
			transfertNumChildrenRootValue = acc.getIf( lightningBoltNode.numChildrenRootOverrideIndex, lightningBoltNode.transfertNumChildrenRootIndex )
			transfertNumChildrenRandRootValue = acc.getIf( lightningBoltNode.numChildrenRootOverrideIndex, lightningBoltNode.transfertNumChildrenRandRootIndex )

			tempCurve = acc.get(lightningBoltNode.inputCurveIndex)
			if tempCurve.isNull():
				sys.stderr.write('the lightninbolt node need at least one curve!\n')
				return
			fnNurbs = OpenMaya.MFnNurbsCurve( tempCurve )
			pointList, cvLength = getPointListFromCurve( self.LP.maxPoints, fnNurbs, childLengthValue )

			outputHandle = acc.get(lightningBoltNode.outputMeshIndex)

		#-------- END collect attributes from the node, now send everything to the lightning processor
			# load the list of points from the curve into the processor
			self.LP.addSeedPointList(pointList)

			# global values
			
			self.LP.setGlobalValue( lightningBoltNode.LM.eGLOBAL.time, timeValue )
			self.LP.setGlobalValue( lightningBoltNode.LM.eGLOBAL.maxGeneration, maxGenerationValue )
			self.LP.setGlobalValue( lightningBoltNode.LM.eGLOBAL.tubeSide, tubeSidesValue )
			self.LP.setGlobalValue( lightningBoltNode.LM.eGLOBAL.vibrationTimeFactor, vibrationTimeFactorValue )
			self.LP.setGlobalValue( lightningBoltNode.LM.eGLOBAL.seedChaos, seedChaosValue )
			self.LP.setGlobalValue( lightningBoltNode.LM.eGLOBAL.seedBranching, seedBranchingValue )
			self.LP.setGlobalValue( lightningBoltNode.LM.eGLOBAL.chaosSecondaryFreqFactor, secondaryChaosFreqFactorValue )
			self.LP.setGlobalValue( lightningBoltNode.LM.eGLOBAL.secondaryChaosMinClamp, secondaryChaosMinClampValue )
			self.LP.setGlobalValue( lightningBoltNode.LM.eGLOBAL.secondaryChaosMaxClamp, secondaryChaosMaxClampValue )
			self.LP.setGlobalValue( lightningBoltNode.LM.eGLOBAL.secondaryChaosMinRemap, secondaryChaosMinRemapValue )
			self.LP.setGlobalValue( lightningBoltNode.LM.eGLOBAL.secondaryChaosMaxRemap, secondaryChaosMaxRemapValue )

			# load the Along Path Values inputs of the node into the processor			
			self.LP.setAPVFactors( lightningBoltNode.LM.eAPBR.radius, radiusValue )
			self.LP.setAPVFactors( lightningBoltNode.LM.eAPBR.intensity, intensityValue )
			self.LP.setAPVFactors( lightningBoltNode.LM.eAPBR.length, cvLength )
			# the corresponding transfert values
			self.LP.setAPVTransfert( lightningBoltNode.LM.eAPBR.radius, transfertRadiusValue, transfertRadiusRootValue )
			self.LP.setAPVTransfert( lightningBoltNode.LM.eAPBR.intensity, transfertIntensityValue, transfertIntensityRootValue )
			self.LP.setAPVTransfert( lightningBoltNode.LM.eAPBR.length, transfertChildLengthValue, transfertChildLengthRootValue )

			# load the Generation inputs
			self.LP.setGENValue( lightningBoltNode.LM.eGEN.chaosFrequency, chaosFrequencyValue )
			self.LP.setGENValue( lightningBoltNode.LM.eGEN.chaosVibration, chaosVibrationValue )
			self.LP.setGENValue( lightningBoltNode.LM.eGEN.chaosTime, timeChaosValue )
			self.LP.setGENValue( lightningBoltNode.LM.eGEN.branchingTime, timeBranchingValue )
			self.LP.setGENValue( lightningBoltNode.LM.eGEN.numChildren, numChildrenValue )
			self.LP.setGENValue( lightningBoltNode.LM.eGEN.numChildrenRand, numChildrenRandValue )
			self.LP.setGENValue( lightningBoltNode.LM.eGEN.chaosOffset, chaosOffsetMultValue )
			self.LP.setGENValue( lightningBoltNode.LM.eGEN.lengthRand, lengthRandValue )
			# the corresponding transfert values
			self.LP.setGENTransfert( lightningBoltNode.LM.eGEN.branchingTime, transfertTimeBranchingValue, transfertTimeBranchingRootValue )
			self.LP.setGENTransfert( lightningBoltNode.LM.eGEN.chaosTime, transfertTimeChaosValue, transfertTimeChaosRootValue )
			self.LP.setGENTransfert( lightningBoltNode.LM.eGEN.chaosOffset, transfertChaosOffsetValue, transfertChaosOffsetRootValue )
			self.LP.setGENTransfert( lightningBoltNode.LM.eGEN.numChildren, transfertNumChildrenValue, transfertNumChildrenRootValue )
			self.LP.setGENTransfert( lightningBoltNode.LM.eGEN.numChildrenRand, transfertNumChildrenRandValue, transfertNumChildrenRandRootValue )
			self.LP.setGENTransfert( lightningBoltNode.LM.eGEN.chaosFrequency, transfertChaosFrequencyValue, transfertChaosFrequencyRootValue )
			self.LP.setGENTransfert( lightningBoltNode.LM.eGEN.chaosVibration, transfertChaosVibrationValue )
			self.LP.setGENTransfert( lightningBoltNode.LM.eGEN.lengthRand, transfertLengthRandValue )
					
			outputData = self.LP.process()
			
			sys.stderr.write('set data\n')
			outputHandle.setMObject(outputData)
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
		self.postConstructorRampInitialize( lightningBoltNode.radiusAlongBranch, [(0,1),(1,0)] )
		self.postConstructorRampInitialize( lightningBoltNode.radiusAlongBranchRoot, [(0,1),(1,0.5)] )
		self.postConstructorRampInitialize( lightningBoltNode.chaosOffsetAlongBranch, [(0,0),(.05,1)] )
		self.postConstructorRampInitialize( lightningBoltNode.chaosOffsetAlongBranchRoot, [(0,0),(.05,1)] )

		self.postConstructorRampInitialize( lightningBoltNode.childLengthAlongBranch, [(0,1),(1,.25)] )
		self.postConstructorRampInitialize( lightningBoltNode.childLengthAlongBranchRoot, [(0,1),(1,.25)] )
		self.postConstructorRampInitialize( lightningBoltNode.intensityAlongPath, [(0,1),(.15,0.8)] )
		self.postConstructorRampInitialize( lightningBoltNode.intensityAlongPathRoot, [(0,1),(1,0.85)] )

		self.postConstructorRampInitialize( lightningBoltNode.elevationAlongBranch, [(0,0.12),(1,0)] )
		self.postConstructorRampInitialize( lightningBoltNode.elevationRandAlongBranch, [(0,0.12),(0,0.3)] )
		self.postConstructorRampInitialize( lightningBoltNode.childProbabilityAlongBranch, [(0,0),(1,1)] )
		

def nodeCreator():
	return OpenMayaMPx.asMPxPtr( lightningBoltNode() )

def nodeInitializer():
	lightningBoltNode.nhlp = MayaAttHelper(lightningBoltNode, eCG.max )

	lightningBoltNode.nhlp.createAtt( 'inputCurve', 'ic', eHlpT.curve, eCG.main )

# Global Values
	lightningBoltNode.nhlp.createAtt( 'detail', 'det', eHlpT.int, eCG.detail, default=6 )

	lightningBoltNode.nhlp.createAtt( 'time', 't', eHlpT.time, eCG.main, default=0.0 )
	lightningBoltNode.nhlp.createAtt( 'tubeSides', 'ts', eHlpT.int, eCG.main, default=4 )
	lightningBoltNode.nhlp.createAtt( 'maxGeneration', 'mg', eHlpT.int, eCG.main, default=3 )
	lightningBoltNode.nhlp.createAtt( 'seedChaos', 'sc', eHlpT.int, eCG.main, default=0 )
	lightningBoltNode.nhlp.createAtt( 'seedBranching', 'sb', eHlpT.int, eCG.main, default=0 )
	lightningBoltNode.nhlp.createAtt( 'vibrationTimeFactor', 'vtf', eHlpT.double, eCG.main, default=150.0 )
	lightningBoltNode.nhlp.createAtt( 'secondaryChaosFreqFactor', 'scff', eHlpT.double, eCG.main, default=4.5 )
	lightningBoltNode.nhlp.createAtt( 'secondaryChaosMinClamp', 'scmnc', eHlpT.double, eCG.main, default=.25 )
	lightningBoltNode.nhlp.createAtt( 'secondaryChaosMaxClamp', 'scmxc', eHlpT.double, eCG.main, default=0.75 )
	lightningBoltNode.nhlp.createAtt( 'secondaryChaosMinRemap', 'scmnr', eHlpT.double, eCG.main, default=0.6 )
	lightningBoltNode.nhlp.createAtt( 'secondaryChaosMaxRemap', 'scmxr', eHlpT.double, eCG.main, default=1.0 )

# Generation Values
	lightningBoltNode.nhlp.createAtt( 'timeChaos', 'tc', eHlpT.double, eCG.main, default=8.0 )
	lightningBoltNode.nhlp.createAtt( 'timeBranching', 'tb', eHlpT.double, eCG.main, default=6.5 )
	lightningBoltNode.nhlp.createAtt( 'chaosFrequency', 'cf', eHlpT.double, eCG.main, default=0.15 )
	lightningBoltNode.nhlp.createAtt( 'chaosVibration', 'cv', eHlpT.double, eCG.main, default=0.15 )
	lightningBoltNode.nhlp.createAtt( 'numChildren', 'nc', eHlpT.int, eCG.main, default=1 )
	lightningBoltNode.nhlp.createAtt( 'numChildrenRand', 'ncr', eHlpT.int, eCG.main, default=3 )
	lightningBoltNode.nhlp.createAtt( 'chaosOffset', 'co', eHlpT.double, eCG.main, default=1.0 )
	lightningBoltNode.nhlp.createAtt( 'lengthRand', 'lr', eHlpT.double, eCG.main, default=0.5 )

	# Generation Transfert Factors
	lightningBoltNode.nhlp.createAtt( 'transfertTimeBranching', 'ttb', eHlpT.double, eCG.main, default=3.0 )
	lightningBoltNode.nhlp.createAtt( 'transfertTimeChaos', 'ttc', eHlpT.double, eCG.main, default=1.0 )
	lightningBoltNode.nhlp.createAtt( 'transfertChaosFrequency', 'tcf', eHlpT.double, eCG.main, default=2.0 )
	lightningBoltNode.nhlp.createAtt( 'transfertChaosVibration', 'tcv', eHlpT.double, eCG.main, default=.6 )
	lightningBoltNode.nhlp.createAtt( 'transfertNumChildren', 'tnc', eHlpT.double, eCG.main, default=0.5 )
	lightningBoltNode.nhlp.createAtt( 'transfertNumChildrenRand', 'tncr', eHlpT.double, eCG.main, default=0.75 )
	lightningBoltNode.nhlp.createAtt( 'transfertChaosOffset', 'tco', eHlpT.double, eCG.main, default=0.6 )
	lightningBoltNode.nhlp.createAtt( 'transfertLengthRand', 'tlr', eHlpT.double, eCG.main, default=1.5 )

# APV Branches
	lightningBoltNode.nhlp.createAtt( 'radiusAlongBranch', 'rab', eHlpT.ramp, eCG.radiusAB )
	lightningBoltNode.nhlp.createAtt( 'radius', 'r', eHlpT.double, eCG.main, default=.15 )
	lightningBoltNode.nhlp.createAtt( 'childLengthAlongBranch', 'clab', eHlpT.ramp, eCG.childLengthAB )
	lightningBoltNode.nhlp.createAtt( 'length', 'l', eHlpT.double, eCG.main, default=1.0 )
	lightningBoltNode.nhlp.createAtt( 'intensityAlongPath', 'ir', eHlpT.ramp, eCG.intensityAB )
	lightningBoltNode.nhlp.createAtt( 'intensity', 'i', eHlpT.double, eCG.main, default=1.0 )

	# APV Branches Transfert Factors
	lightningBoltNode.nhlp.createAtt( 'transfertRadius', 'tr', eHlpT.double, eCG.main, default=0.8 )
	lightningBoltNode.nhlp.createAtt( 'transfertLength', 'tl', eHlpT.double, eCG.main, default=0.6 )
	lightningBoltNode.nhlp.createAtt( 'transfertIntensity', 'ti', eHlpT.double, eCG.main, default=1.0 )

# APV Specials
	lightningBoltNode.nhlp.createAtt( 'chaosOffsetAlongBranch', 'coab', eHlpT.ramp, eCG.chaosOffsetAB )

	# elevation 1 = 180 degree
	lightningBoltNode.nhlp.createAtt( 'elevationAlongBranch', 'eab', eHlpT.ramp, eCG.elevationAB )
	lightningBoltNode.nhlp.createAtt( 'elevationRandAlongBranch', 'erab', eHlpT.ramp, eCG.elevationRandAB )
	lightningBoltNode.nhlp.createAtt( 'childProbabilityAlongBranch', 'cpab', eHlpT.ramp, eCG.childProbabilityAB )

# Root Overrides
	# Generation Transfert Overrides
	lightningBoltNode.nhlp.createAtt( 'timeRootOverride', 'tro', eHlpT.bool, eCG.main, default=False )
	lightningBoltNode.nhlp.createAtt( 'transfertTimeChaosRoot', 'ttsrt', eHlpT.double, eCG.main, default=1.0 )
	lightningBoltNode.nhlp.createAtt( 'transfertTimeBranchingRoot', 'ttbrt', eHlpT.double, eCG.main, default=1.0 )

	lightningBoltNode.nhlp.createAtt( 'chaosOffsetRootOverride', 'coro', eHlpT.bool, eCG.chaosOffsetABRoot, default=False )
	lightningBoltNode.nhlp.createAtt( 'chaosOffsetAlongBranchRoot', 'orrt', eHlpT.ramp, eCG.chaosOffsetABRoot )
	lightningBoltNode.nhlp.createAtt( 'transfertChaosOffsetRoot', 'tor', eHlpT.double, eCG.main, default=1.0 )
	lightningBoltNode.nhlp.createAtt( 'transfertChaosFrequencyRoot', 'tcfr', eHlpT.double, eCG.main, default=1.0 )

	lightningBoltNode.nhlp.createAtt( 'numChildrenRootOverride', 'ncro', eHlpT.bool, eCG.main, default=0 )
	lightningBoltNode.nhlp.createAtt( 'transfertNumChildrenRoot', 'tncrt', eHlpT.double, eCG.main, default=1.0 )
	lightningBoltNode.nhlp.createAtt( 'transfertNumChildrenRandRoot', 'tncrrt', eHlpT.double, eCG.main, default=1.0 )


	# APV Special overrides
	lightningBoltNode.nhlp.createAtt( 'elevationRootOverride', 'ero', eHlpT.bool, [eCG.elevationABRoot, eCG.elevationRandABRoot], default=False )
	lightningBoltNode.nhlp.createAtt( 'elevationAlongBranchRoot', 'errt', eHlpT.ramp, eCG.elevationABRoot )	
	lightningBoltNode.nhlp.createAtt( 'elevationRandAlongBranchRoot', 'errrt', eHlpT.ramp, eCG.elevationRandABRoot )

	# APV Branches overrides
	lightningBoltNode.nhlp.createAtt( 'radiusRootOverride', 'rro', eHlpT.bool, eCG.radiusABRoot, default=True )
	lightningBoltNode.nhlp.createAtt( 'radiusAlongBranchRoot', 'rrrt', eHlpT.ramp, eCG.radiusABRoot )
	lightningBoltNode.nhlp.createAtt( 'transfertRadiusRoot', 'trr', eHlpT.double, eCG.main, default=0.45 )
	
	lightningBoltNode.nhlp.createAtt( 'lengthRootOverride', 'lro', eHlpT.bool, eCG.childLengthABRoot, default=False )
	lightningBoltNode.nhlp.createAtt( 'childLengthAlongBranchRoot', 'clapr', eHlpT.ramp, eCG.childLengthABRoot )
	lightningBoltNode.nhlp.createAtt( 'transfertLengthRoot', 'tlro', eHlpT.double, eCG.main, default=1.0 )
	
	lightningBoltNode.nhlp.createAtt( 'intensityRootOverride', 'iro', eHlpT.bool, eCG.intensityABRoot, default=True )
	lightningBoltNode.nhlp.createAtt( 'intensityAlongPathRoot', 'iapr', eHlpT.ramp, eCG.intensityABRoot )
	lightningBoltNode.nhlp.createAtt( 'transfertIntensityRoot', 'tir', eHlpT.double, eCG.main, default=1.0 )

# OUTPUTS
	lightningBoltNode.nhlp.createAtt( 'outputMesh', 'om', eHlpT.mesh, eCG.main, isInput=False )

	lightningBoltNode.nhlp.addGroupDependencies( eCG.radiusAB, [eCG.detail] )
	lightningBoltNode.nhlp.addGroupDependencies( eCG.childLengthAB, [eCG.detail] )
	lightningBoltNode.nhlp.addGroupDependencies( eCG.intensityAB, [eCG.detail] )
	lightningBoltNode.nhlp.addGroupDependencies( eCG.chaosOffsetAB, [eCG.detail] )
	lightningBoltNode.nhlp.addGroupDependencies( eCG.elevationAB, [eCG.detail] )
	lightningBoltNode.nhlp.addGroupDependencies( eCG.elevationRandAB, [eCG.detail] )
	lightningBoltNode.nhlp.addGroupDependencies( eCG.childProbabilityAB, [eCG.detail] )
	
	lightningBoltNode.nhlp.addGroupDependencies( eCG.radiusABRoot, [eCG.detail] )
	lightningBoltNode.nhlp.addGroupDependencies( eCG.childLengthABRoot, [eCG.detail] )
	lightningBoltNode.nhlp.addGroupDependencies( eCG.intensityABRoot, [eCG.detail] )
	lightningBoltNode.nhlp.addGroupDependencies( eCG.chaosOffsetABRoot, [eCG.detail] )
	lightningBoltNode.nhlp.addGroupDependencies( eCG.elevationABRoot, [eCG.detail] )
	lightningBoltNode.nhlp.addGroupDependencies( eCG.elevationRandABRoot, [eCG.detail] )
	
	lightningBoltNode.nhlp.addGroupDependencies( eCG.main, [ eCG.radiusAB, eCG.childLengthAB, eCG.intensityAB, eCG.chaosOffsetAB, eCG.elevationAB, eCG.elevationRandAB, eCG.childProbabilityAB, eCG.radiusABRoot, eCG.childLengthABRoot, eCG.intensityABRoot, eCG.chaosOffsetABRoot, eCG.elevationABRoot, eCG.elevationRandABRoot ] )

	lightningBoltNode.nhlp.finalizeAttributeInitialization()
	sys.stderr.write('end of initialize\n')


def cleanupClass():
	delattr(lightningBoltNode,'LM')
	lightningBoltNode.nhlp.cleanUp()
	delattr(lightningBoltNode,'nhlp')

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
