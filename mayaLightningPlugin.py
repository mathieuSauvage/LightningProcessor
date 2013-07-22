#!/usr/bin/python
# -*- coding: iso-8859-1 -*-


'''

import pymel.core as pm

pm.loadPlugin('/Users/mathieu/Documents/IMPORTANT/Perso/Prog/python/lightningPlugin/mayaLightningPlugin.py')

lightNode = pm.createNode('lightningbolt')
meshNode = pm.createNode('mesh')
pm.connectAttr( lightNode+'.outputMesh', meshNode+'.inMesh')
pm.connectAttr( 'curveShape1.worldSpace[0]', lightNode+'.inputCurve')

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
		if type is OpenMaya.MFnNumericData.kBoolean :
			return tempData.asBool()
		if type is OpenMaya.MFnUnitAttribute.kTime :
			return tempData.asTime()
		if type is OpenMaya.MFnNumericData.kInt :
			return tempData.asInt()
		if type is OpenMaya.MFnData.kNurbsCurve :
			return tempData.asNurbsCurveTransformed()
		if type is "vectorOfDouble" :
			return OpenMaya.MVector( tempData.asVector() )

		raise 'type not handled in getAttValueOrHdl!'


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

	mObj = meshFn.create(numVertices, numFaces, Mpoints , MfacesCount, MfaceConnects, outData)

	# vertexColors
	scrUtil.createFromList( resultIntensityColors, len(resultIntensityColors))
	MColors = OpenMaya.MColorArray( scrUtil.asDouble4Ptr(), numVertices )

	scrUtil.createFromList( range(numVertices) , numVertices )
	MVertexIds = OpenMaya.MIntArray( scrUtil.asIntPtr(), numVertices )

	name = 'lightningIntensity'
	fName = meshFn.createColorSetDataMesh(name)

	#meshFn.setCurrentColorSetName(name)

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

def setupAPVInputFromRamp( arrays, ramp, rampRoot=None ):
	array, arrayRoot = arrays

	if array.size <1 :
		raise 'array size must be more than 1'

	fillArrayFromRampAtt( array, ramp)
	if rampRoot is None :
		arrayRoot[:] = array
		return
	fillArrayFromRampAtt( arrayRoot, rampRoot)

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

# class to help having a cleaner access to node's attribute values
class ATTAccessor:
	def __init__( self, helper, node, data ):
		self.helper = helper
		self.node = node
		self.data = data

	def get( self, att ):
		return self.helper.getAttValueOrHdl( att, self.node, self.data )

	def getIf( self, attTest, att ):
		boolVal = self.get( attTest )
		result = None
		if boolVal:
			result = self.get( att )
		return result

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
	LGHTAEmakeLargeRamp( $nodeAttr,1,1,0,0, 1 );
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

		editorTemplate -beginLayout "Global Processor parameters" -collapse 1;
			editorTemplate -addControl "tubeSides";
			editorTemplate -addControl "maxGeneration";
			editorTemplate -addControl "detail";
			editorTemplate -addControl "seedChaos";
			editorTemplate -addControl "seedBranching";
		editorTemplate -endLayout;

		editorTemplate -beginLayout "⟦ Length Attributes ⟧" -collapse 1;
			editorTemplate -callCustom "LGHTAELineBaseFloatTransfertLayout" "LGHTAELineBaseTransfertLayout_Replace" "length" "transfertLength";
			editorTemplate -callCustom "LGHTAELineBaseFloatTransfertLayout" "LGHTAELineBaseTransfertLayout_Replace" "lengthRand" "transfertLengthRand";
			editorTemplate -callCustom "LGHTAERampControl" "LGHTAERampControl_Replace" "childLengthAlongPath";
			editorTemplate -callCustom "LGHTAEOverrideLayoutLength" "LGHTAEOverrideLayoutLength_Replace" "lengthRootOverride" "transfertLengthRoot" "childLengthAlongPathRoot";


			editorTemplate -s "length";
			editorTemplate -s "transfertLength";
			editorTemplate -s "lengthRand";
			editorTemplate -s "transfertLengthRand";
			editorTemplate -s "childLengthAlongPath";
			editorTemplate -s "lengthRootOverride";
			editorTemplate -s "transfertLengthRoot";
			editorTemplate -s "childLengthAlongPathRoot";
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
			editorTemplate -callCustom "LGHTAELineBaseFloatTransfertLayout" "LGHTAELineBaseTransfertLayout_Replace" "timeChaos" "transfertTimeShape";
			editorTemplate -addControl "vibrationTimeFactor";
			editorTemplate -callCustom "LGHTAELineBaseFloatTransfertLayout" "LGHTAELineBaseTransfertLayout_Replace" "timeBranching" "transfertTimeBranching";
			editorTemplate -callCustom "LGHTAEOverrideLayoutTime" "LGHTAEOverrideLayoutTime_Replace" "timeRootOverride" "transfertTimeShapeRoot" "transfertTimeBranchingRoot";				
			editorTemplate -s "timeChaos";
			editorTemplate -s "vibrationTimeFactor";
			editorTemplate -s "transfertTimeShape";
			editorTemplate -s "timeBranching";
			editorTemplate -s "transfertTimeBranching";
			editorTemplate -s "timeRootOverride";
			editorTemplate -s "transfertTimeShapeRoot";
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

	# let's have the lightningModule here as a static
	lightningModule = loadGeneralLightningScript()
	# shortcuts
	LPM = lightningModule.lightningBoltProcessor

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
		self.LP = lightningBoltNode.LPM(mayaLightningMesher)

	def compute(self, plug, data):

		if plug == lightningBoltNode.samplingDummyOut.mObject : # outDummyTrigger to Sample Ramps
			sys.stderr.write('compute dummy\n')

			thisNode = self.thisMObject()
			acc = ATTAccessor( lightningBoltNode.hlp, thisNode, data )

			# Basic Ramp Attribute to Sample

			tempDetail = acc.get(lightningBoltNode.detail)
			# ***** initialisation of the sizes of arrays in the Processor (all Along Path arrays depend on detail)
			self.LP.initialize(tempDetail)

			tempRadiusRamp = acc.get(lightningBoltNode.radiusAlongBranch)
			tempIntensityRamp = acc.get(lightningBoltNode.intensityAlongPath)
			tempChaosOffsetRamp = acc.get(lightningBoltNode.chaosOffsetAlongBranch)
			tempChildLengthRamp = acc.get(lightningBoltNode.childLengthAlongPath)
			tempElevationRamp = acc.get(lightningBoltNode.elevationAlongBranch)
			tempElevationRandRamp = acc.get(lightningBoltNode.elevationRandAlongBranch)
			tempChildProbabilityRamp = acc.get(lightningBoltNode.childProbabilityAlongBranch)

			tempRadiusRampRoot = acc.getIf( lightningBoltNode.radiusRootOverride, lightningBoltNode.radiusAlongBranchRoot )
			tempChildLengthRampRoot = acc.getIf( lightningBoltNode.lengthRootOverride, lightningBoltNode.childLengthAlongPathRoot )
			tempIntensityRampRoot = acc.getIf( lightningBoltNode.intensityRootOverride, lightningBoltNode.intensityAlongPathRoot )
			tempElevationRampRoot = acc.getIf( lightningBoltNode.elevationRootOverride, lightningBoltNode.elevationAlongBranchRoot )
			tempElevationRandRampRoot = acc.getIf( lightningBoltNode.elevationRootOverride, lightningBoltNode.elevationRandAlongBranchRoot )
			tempChaosOffsetRampRoot = acc.getIf( lightningBoltNode.chaosOffsetRootOverride, lightningBoltNode.chaosOffsetAlongBranchRoot )

			setupAPVInputFromRamp( self.LP.getAPBR( lightningBoltNode.LPM.eAPBR.radius ), tempRadiusRamp, tempRadiusRampRoot )
			setupAPVInputFromRamp( self.LP.getAPBR( lightningBoltNode.LPM.eAPBR.intensity ), tempIntensityRamp, tempIntensityRampRoot )
			setupAPVInputFromRamp( self.LP.getAPBR( lightningBoltNode.LPM.eAPBR.length ), tempChildLengthRamp, tempChildLengthRampRoot )

			setupAPVInputFromRamp( self.LP.getAPSPE( lightningBoltNode.LPM.eAPSPE.chaosOffset ), tempChaosOffsetRamp, tempChaosOffsetRampRoot )
			setupAPVInputFromRamp( self.LP.getAPSPE( lightningBoltNode.LPM.eAPSPE.elevation ), tempElevationRamp, tempElevationRampRoot )
			setupAPVInputFromRamp( self.LP.getAPSPE( lightningBoltNode.LPM.eAPSPE.elevationRand ), tempElevationRandRamp, tempElevationRandRampRoot )
			setupAPVInputFromRamp( self.LP.getAPSPE( lightningBoltNode.LPM.eAPSPE.childProbability ), tempChildProbabilityRamp )

			outputHandle = lightningBoltNode.hlp.getAttValueOrHdl( lightningBoltNode.samplingDummyOut, thisNode,data)
			outputHandle.setInt(1)
			data.setClean(plug)

		elif plug == lightningBoltNode.outputMesh.mObject : # outMesh
			sys.stderr.write('compute mesh\n')

			thisNode = self.thisMObject()
			acc = ATTAccessor( lightningBoltNode.hlp, thisNode, data )

			# global values
			tempTubeSides = acc.get( lightningBoltNode.tubeSides)
			tempMaxGeneration = acc.get( lightningBoltNode.maxGeneration)
			tempVibrationTimeFactor = acc.get( lightningBoltNode.vibrationTimeFactor)
			tempSecondaryChaosFreqFactor = acc.get( lightningBoltNode.secondaryChaosFreqFactor )

			tempSecondaryChaosMinClamp = acc.get( lightningBoltNode.secondaryChaosMinClamp )
			tempSecondaryChaosMaxClamp = acc.get( lightningBoltNode.secondaryChaosMaxClamp )
			tempSecondaryChaosMinRemap = acc.get( lightningBoltNode.secondaryChaosMinRemap )
			tempSecondaryChaosMaxRemap = acc.get( lightningBoltNode.secondaryChaosMaxRemap )
			

			tempSeedChaos = acc.get( lightningBoltNode.seedChaos)
			tempSeedBranching = acc.get( lightningBoltNode.seedBranching)

			# generation values
			tempChaosFrequency = acc.get( lightningBoltNode.chaosFrequency)
			tempChaosVibration = acc.get( lightningBoltNode.chaosVibration)
			tempNumChildren = acc.get( lightningBoltNode.numChildren)
			tempNumChildrenRand = acc.get( lightningBoltNode.numChildrenRand)
			tempTimeShape = acc.get( lightningBoltNode.timeChaos)
			tempTimeBranching = acc.get( lightningBoltNode.timeBranching)
			tempChaosOffsetMult = acc.get( lightningBoltNode.chaosOffset)
			tempLengthRand = acc.get( lightningBoltNode.lengthRand)


			# Along Path branches values
			tempRadiusMult = acc.get( lightningBoltNode.radius)
			tempIntensityMult = acc.get( lightningBoltNode.intensity)
			tempChildLengthMult = acc.get( lightningBoltNode.length)

			# transfert values
			tempTransfertRadius = acc.get( lightningBoltNode.transfertRadius)
			tempTransfertChildLength = acc.get( lightningBoltNode.transfertLength)
			tempTransfertIntensity = acc.get( lightningBoltNode.transfertIntensity)

			tempTransfertChaosOffset = acc.get( lightningBoltNode.transfertChaosOffset)
			tempTransfertTimeBranching = acc.get( lightningBoltNode.transfertTimeBranching)
			tempTransfertTimeShape = acc.get( lightningBoltNode.transfertTimeShape)
			tempTransfertNumChildren = acc.get( lightningBoltNode.transfertNumChildren)
			tempTransfertNumChildrenRand = acc.get( lightningBoltNode.transfertNumChildrenRand)
			tempTransfertChaosFrequency = acc.get( lightningBoltNode.transfertChaosFrequency)
			tempTransfertChaosVibration = acc.get( lightningBoltNode.transfertChaosVibration)
			tempTransfertLengthRand = acc.get( lightningBoltNode.transfertLengthRand)


			# Root overrides
			tempTransfertRadiusRoot = acc.getIf( lightningBoltNode.radiusRootOverride, lightningBoltNode.transfertRadiusRoot )
			tempTransfertChildLengthRoot = acc.getIf( lightningBoltNode.lengthRootOverride, lightningBoltNode.transfertLengthRoot )
			tempTransfertIntensityRoot = acc.getIf( lightningBoltNode.intensityRootOverride, lightningBoltNode.transfertIntensityRoot )
			tempTransfertTimeShapeRoot = acc.getIf( lightningBoltNode.timeRootOverride, lightningBoltNode.transfertTimeShapeRoot )
			tempTransfertTimeBranchingRoot = acc.getIf( lightningBoltNode.timeRootOverride, lightningBoltNode.transfertTimeBranchingRoot )
			tempTransfertChaosOffsetRoot = acc.getIf( lightningBoltNode.chaosOffsetRootOverride, lightningBoltNode.transfertChaosOffsetRoot )
			tempTransfertNumChildrenRoot = acc.getIf( lightningBoltNode.numChildrenRootOverride, lightningBoltNode.transfertNumChildrenRoot )
			tempTransfertNumChildrenRandRoot = acc.getIf( lightningBoltNode.numChildrenRootOverride, lightningBoltNode.transfertNumChildrenRandRoot )
			tempTransfertChaosFrequencyRoot = acc.getIf( lightningBoltNode.chaosOffsetRootOverride, lightningBoltNode.transfertChaosFrequencyRoot )

			# force evaluation if needed of ramp samples (this will trigger the plug for outDummyAtt of the compute)
			plugDummy = OpenMaya.MPlug( thisNode, lightningBoltNode.samplingDummyOut.mObject) 
			triggerSampling = plugDummy.asInt()


			outputHandle = acc.get(lightningBoltNode.outputMesh)

			tempCurve = acc.get(lightningBoltNode.inputCurve)
			if tempCurve.isNull():
				sys.stderr.write('there is no curve!\n')
				data.setClean(plug)
				return

			fnNurbs = OpenMaya.MFnNurbsCurve( tempCurve )
			pointList, cvLength = getPointListFromCurve( self.LP.maxPoints, fnNurbs, tempChildLengthMult )

			#sys.stderr.write('child length '+str(tempChildLengthMult*cvLength)+'\n')

			# load the Along Path Values inputs of the node into the processor			
			self.LP.setAPVFactors( lightningBoltNode.LPM.eAPBR.radius, tempRadiusMult )
			self.LP.setAPVFactors( lightningBoltNode.LPM.eAPBR.intensity, tempIntensityMult )
			self.LP.setAPVFactors( lightningBoltNode.LPM.eAPBR.length, cvLength )

			# load the Generation inputs
			self.LP.setGENValue( lightningBoltNode.LPM.eGEN.chaosFrequency, tempChaosFrequency )
			self.LP.setGENValue( lightningBoltNode.LPM.eGEN.chaosVibration, tempChaosVibration )
			self.LP.setGENValue( lightningBoltNode.LPM.eGEN.chaosTime, tempTimeShape.value() )
			self.LP.setGENValue( lightningBoltNode.LPM.eGEN.branchingTime, tempTimeBranching.value() )
			self.LP.setGENValue( lightningBoltNode.LPM.eGEN.numChildren, tempNumChildren )
			self.LP.setGENValue( lightningBoltNode.LPM.eGEN.numChildrenRand, tempNumChildrenRand )
			self.LP.setGENValue( lightningBoltNode.LPM.eGEN.chaosOffset, tempChaosOffsetMult )
			self.LP.setGENValue( lightningBoltNode.LPM.eGEN.lengthRand, tempLengthRand )

			# load the Generation Transfert Factors
			self.LP.setGENTransfert( lightningBoltNode.LPM.eGEN.branchingTime, tempTransfertTimeBranching, tempTransfertTimeBranchingRoot )
			self.LP.setGENTransfert( lightningBoltNode.LPM.eGEN.chaosTime, tempTransfertTimeShape, tempTransfertTimeShapeRoot )
			self.LP.setGENTransfert( lightningBoltNode.LPM.eGEN.chaosOffset, tempTransfertChaosOffset, tempTransfertChaosOffsetRoot )
			self.LP.setGENTransfert( lightningBoltNode.LPM.eGEN.numChildren, tempTransfertNumChildren, tempTransfertNumChildrenRoot )
			self.LP.setGENTransfert( lightningBoltNode.LPM.eGEN.numChildrenRand, tempTransfertNumChildrenRand, tempTransfertNumChildrenRandRoot )
			self.LP.setGENTransfert( lightningBoltNode.LPM.eGEN.chaosFrequency, tempTransfertChaosFrequency,tempTransfertChaosFrequencyRoot )
			self.LP.setGENTransfert( lightningBoltNode.LPM.eGEN.chaosVibration, tempTransfertChaosVibration )
			self.LP.setGENTransfert( lightningBoltNode.LPM.eGEN.lengthRand, tempTransfertLengthRand )


			self.LP.setAPVTransfert( lightningBoltNode.LPM.eAPBR.radius, tempTransfertRadius, tempTransfertRadiusRoot )
			self.LP.setAPVTransfert( lightningBoltNode.LPM.eAPBR.length, tempTransfertChildLength, tempTransfertChildLengthRoot )
			self.LP.setAPVTransfert( lightningBoltNode.LPM.eAPBR.intensity, tempTransfertIntensity, tempTransfertIntensityRoot )
			

			#self.LP.resetProcessor()
			self.LP.addSeedPointList(pointList)

			self.LP.secondaryChaosMinClamp = tempSecondaryChaosMinClamp
			self.LP.secondaryChaosMaxClamp = tempSecondaryChaosMaxClamp
			self.LP.secondaryChaosMinRemap = tempSecondaryChaosMinRemap
			self.LP.secondaryChaosMaxRemap = tempSecondaryChaosMaxRemap

			outputData = self.LP.process(tempMaxGeneration, tempTubeSides, tempVibrationTimeFactor, tempSecondaryChaosFreqFactor, tempSeedChaos, tempSeedBranching)

			#sys.stderr.write('end of compute\n')

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
		self.postConstructorRampInitialize( lightningBoltNode.radiusAlongBranch.mObject, [(0,1),(1,0)] )
		self.postConstructorRampInitialize( lightningBoltNode.radiusAlongBranchRoot.mObject, [(0,1),(1,0)] )
		self.postConstructorRampInitialize( lightningBoltNode.chaosOffsetAlongBranch.mObject, [(0,0),(.05,1)] )
		self.postConstructorRampInitialize( lightningBoltNode.chaosOffsetAlongBranchRoot.mObject, [(0,0),(.05,1)] )

		self.postConstructorRampInitialize( lightningBoltNode.childLengthAlongPath.mObject, [(0,1),(1,.25)] )
		self.postConstructorRampInitialize( lightningBoltNode.childLengthAlongPathRoot.mObject, [(0,1),(1,.25)] )
		self.postConstructorRampInitialize( lightningBoltNode.intensityAlongPath.mObject, [(0,1),(1,0.5)] )
		self.postConstructorRampInitialize( lightningBoltNode.intensityAlongPathRoot.mObject, [(0,1),(1,0.5)] )

		self.postConstructorRampInitialize( lightningBoltNode.elevationAlongBranch.mObject, [(0,0.5),(1,.15)] )
		self.postConstructorRampInitialize( lightningBoltNode.elevationRandAlongBranch.mObject, [(0,1)] )
		self.postConstructorRampInitialize( lightningBoltNode.childProbabilityAlongBranch.mObject, [(0,0),(1,1)] )

def nodeCreator():
	return OpenMayaMPx.asMPxPtr( lightningBoltNode() )

def nodeInitializer():
	unitAttr = OpenMaya.MFnUnitAttribute()
	typedAttr = OpenMaya.MFnTypedAttribute()
	numAttr = OpenMaya.MFnNumericAttribute()
	#compAttr = OpenMaya.MFnCompoundAttribute()

	lightningBoltNode.hlp = attCreationHlp(lightningBoltNode)

	lightningBoltNode.hlp.createAtt( name = "inputCurve", fn=typedAttr, shortName="ic", type=OpenMaya.MFnData.kNurbsCurve, exceptAffectList=['samplingDummyOut'] )	

# Processor Values
	lightningBoltNode.hlp.createAtt( name = "detail", fn=numAttr, shortName="det", type=OpenMaya.MFnNumericData.kInt, default=lightningBoltNode.defaultDetailValue )
	lightningBoltNode.hlp.createAtt( name = "tubeSides", fn=numAttr, shortName="tus", type=OpenMaya.MFnNumericData.kInt, default=lightningBoltNode.defaultTubeSides, exceptAffectList=['samplingDummyOut'] )
	lightningBoltNode.hlp.createAtt( name = "maxGeneration", fn=numAttr, shortName="mg", type=OpenMaya.MFnNumericData.kInt, default=lightningBoltNode.defaultMaxGeneration, exceptAffectList=['samplingDummyOut'] )	
	lightningBoltNode.hlp.createAtt( name = "seedChaos", fn=numAttr, shortName="ss", type=OpenMaya.MFnNumericData.kInt, default=0.0, exceptAffectList=['samplingDummyOut'] )	
	lightningBoltNode.hlp.createAtt( name = "seedBranching", fn=numAttr, shortName="sb", type=OpenMaya.MFnNumericData.kInt, default=0.0, exceptAffectList=['samplingDummyOut'] )	
	lightningBoltNode.hlp.createAtt( name = "vibrationTimeFactor", fn=numAttr, shortName="vtf", type=OpenMaya.MFnNumericData.kDouble, default=150.0, exceptAffectList=['samplingDummyOut'] )
	lightningBoltNode.hlp.createAtt( name = "secondaryChaosFreqFactor", fn=numAttr, shortName="scff", type=OpenMaya.MFnNumericData.kDouble, default=4.0, exceptAffectList=['samplingDummyOut'] )
	lightningBoltNode.hlp.createAtt( name = "secondaryChaosMinClamp", fn=numAttr, shortName="scmnc", type=OpenMaya.MFnNumericData.kDouble, default=0.25, exceptAffectList=['samplingDummyOut'] )
	lightningBoltNode.hlp.createAtt( name = "secondaryChaosMaxClamp", fn=numAttr, shortName="scmxc", type=OpenMaya.MFnNumericData.kDouble, default=0.75, exceptAffectList=['samplingDummyOut'] )
	lightningBoltNode.hlp.createAtt( name = "secondaryChaosMinRemap", fn=numAttr, shortName="scmncr", type=OpenMaya.MFnNumericData.kDouble, default=0.2, exceptAffectList=['samplingDummyOut'] )
	lightningBoltNode.hlp.createAtt( name = "secondaryChaosMaxRemap", fn=numAttr, shortName="scmxcr", type=OpenMaya.MFnNumericData.kDouble, default=0.8, exceptAffectList=['samplingDummyOut'] )

# Generation Values
	lightningBoltNode.hlp.createAtt( name = "timeChaos", fn=unitAttr, shortName="ts", type=OpenMaya.MFnUnitAttribute.kTime, default=0.0, exceptAffectList=['samplingDummyOut'] )
	lightningBoltNode.hlp.createAtt( name = "timeBranching", fn=unitAttr, shortName="tb", type=OpenMaya.MFnUnitAttribute.kTime, default=0.0, exceptAffectList=['samplingDummyOut'] )
	lightningBoltNode.hlp.createAtt( name = "chaosFrequency", fn=numAttr, shortName="sf", type=OpenMaya.MFnNumericData.kDouble, default=0.2, exceptAffectList=['samplingDummyOut'] )
	lightningBoltNode.hlp.createAtt( name = "chaosVibration", fn=numAttr, shortName="cv", type=OpenMaya.MFnNumericData.kDouble, default=0.35, exceptAffectList=['samplingDummyOut'] )
	lightningBoltNode.hlp.createAtt( name = "numChildren", fn=numAttr, shortName="nc", type=OpenMaya.MFnNumericData.kInt, default=lightningBoltNode.defaultNumChildren, exceptAffectList=['samplingDummyOut'] )	
	lightningBoltNode.hlp.createAtt( name = "numChildrenRand", fn=numAttr, shortName="ncr", type=OpenMaya.MFnNumericData.kInt, default=lightningBoltNode.defaultNumChildrenRand, exceptAffectList=['samplingDummyOut'] )
	lightningBoltNode.hlp.createAtt( name = "chaosOffset", fn=numAttr, shortName="om", type=OpenMaya.MFnNumericData.kDouble, default=1.0, exceptAffectList=['samplingDummyOut'] )
	lightningBoltNode.hlp.createAtt( name = "lengthRand", fn=numAttr, shortName="lr", type=OpenMaya.MFnNumericData.kDouble, default=1.5, exceptAffectList=['samplingDummyOut'] )


	# Generation Transfert Factors
	lightningBoltNode.hlp.createAtt( name = "transfertTimeBranching", fn=numAttr, shortName="ttb", type=OpenMaya.MFnNumericData.kDouble, default=3.0, exceptAffectList=['samplingDummyOut'] )
	lightningBoltNode.hlp.createAtt( name = "transfertTimeShape", fn=numAttr, shortName="tts", type=OpenMaya.MFnNumericData.kDouble, default=1.0, exceptAffectList=['samplingDummyOut'] )	
	lightningBoltNode.hlp.createAtt( name = "transfertChaosFrequency", fn=numAttr, shortName="tsf", type=OpenMaya.MFnNumericData.kDouble, default=1.5, exceptAffectList=['samplingDummyOut'] )
	lightningBoltNode.hlp.createAtt( name = "transfertChaosVibration", fn=numAttr, shortName="tcv", type=OpenMaya.MFnNumericData.kDouble, default=0.75, exceptAffectList=['samplingDummyOut'] )
	lightningBoltNode.hlp.createAtt( name = "transfertNumChildren", fn=numAttr, shortName="tnc", type=OpenMaya.MFnNumericData.kDouble, default=1.0, exceptAffectList=['samplingDummyOut'] )	
	lightningBoltNode.hlp.createAtt( name = "transfertNumChildrenRand", fn=numAttr, shortName="tncr", type=OpenMaya.MFnNumericData.kDouble, default=1.0, exceptAffectList=['samplingDummyOut'] )	
	lightningBoltNode.hlp.createAtt( name = "transfertChaosOffset", fn=numAttr, shortName="to", type=OpenMaya.MFnNumericData.kDouble, default=0.5, exceptAffectList=['samplingDummyOut'] )
	lightningBoltNode.hlp.createAtt( name = "transfertLengthRand", fn=numAttr, shortName="tlr", type=OpenMaya.MFnNumericData.kDouble, default=1.5, exceptAffectList=['samplingDummyOut'] )


# APV Branches
	lightningBoltNode.hlp.createAtt( name = "radiusAlongBranch", fn=OpenMaya.MRampAttribute, shortName="rr", type="Ramp" )
	lightningBoltNode.hlp.createAtt( name = "radius", fn=numAttr, shortName="rm", type=OpenMaya.MFnNumericData.kDouble, default=0.25, exceptAffectList=['samplingDummyOut'] )	
	
	lightningBoltNode.hlp.createAtt( name = "childLengthAlongPath", fn=OpenMaya.MRampAttribute, shortName="clr", type="Ramp" )
	lightningBoltNode.hlp.createAtt( name = "length", fn=numAttr, shortName="clm", type=OpenMaya.MFnNumericData.kDouble, default=1.0, exceptAffectList=['samplingDummyOut'] )	
	
	lightningBoltNode.hlp.createAtt( name = "intensityAlongPath", fn=OpenMaya.MRampAttribute, shortName="ir", type="Ramp" )
	lightningBoltNode.hlp.createAtt( name = "intensity", fn=numAttr, shortName="im", type=OpenMaya.MFnNumericData.kDouble, default=1.0, exceptAffectList=['samplingDummyOut'] )	

	# APV Branches Transfert Factors
	lightningBoltNode.hlp.createAtt( name = "transfertRadius", fn=numAttr, shortName="tr", type=OpenMaya.MFnNumericData.kDouble, default=0.6, exceptAffectList=['samplingDummyOut'] )	
	lightningBoltNode.hlp.createAtt( name = "transfertLength", fn=numAttr, shortName="tcl", type=OpenMaya.MFnNumericData.kDouble, default=0.7, exceptAffectList=['samplingDummyOut'] )	
	lightningBoltNode.hlp.createAtt( name = "transfertIntensity", fn=numAttr, shortName="ti", type=OpenMaya.MFnNumericData.kDouble, default=1.0, exceptAffectList=['samplingDummyOut'] )	

# APV Specials
	lightningBoltNode.hlp.createAtt( name = "chaosOffsetAlongBranch", fn=OpenMaya.MRampAttribute, shortName="or", type="Ramp" )	
	# elevation 1 = 180 degree
	lightningBoltNode.hlp.createAtt( name = "elevationAlongBranch", fn=OpenMaya.MRampAttribute, shortName="er", type="Ramp" )
	lightningBoltNode.hlp.createAtt( name = "elevationRandAlongBranch", fn=OpenMaya.MRampAttribute, shortName="err", type="Ramp" )
	lightningBoltNode.hlp.createAtt( name = "childProbabilityAlongBranch", fn=OpenMaya.MRampAttribute, shortName="cpr", type="Ramp" )

# Root Overrides
	# Generation Transfert Overrides
	lightningBoltNode.hlp.createAtt( name = "timeRootOverride", fn=numAttr, shortName="tro", type=OpenMaya.MFnNumericData.kBoolean, default=0 )	
	lightningBoltNode.hlp.createAtt( name = "transfertTimeShapeRoot", fn=numAttr, shortName="ttsrt", type=OpenMaya.MFnNumericData.kDouble, default=1.0, exceptAffectList=['samplingDummyOut'] )
	lightningBoltNode.hlp.createAtt( name = "transfertTimeBranchingRoot", fn=numAttr, shortName="ttbrt", type=OpenMaya.MFnNumericData.kDouble, default=1.0, exceptAffectList=['samplingDummyOut'] )

	lightningBoltNode.hlp.createAtt( name = "chaosOffsetRootOverride", fn=numAttr, shortName="oro", type=OpenMaya.MFnNumericData.kBoolean, default=0 )	
	lightningBoltNode.hlp.createAtt( name = "transfertChaosOffsetRoot", fn=numAttr, shortName="tor", type=OpenMaya.MFnNumericData.kDouble, default=1.0, exceptAffectList=['samplingDummyOut'] )	
	lightningBoltNode.hlp.createAtt( name = "transfertChaosFrequencyRoot", fn=numAttr, shortName="tsfr", type=OpenMaya.MFnNumericData.kDouble, default=1.0, exceptAffectList=['samplingDummyOut'] )	
	lightningBoltNode.hlp.createAtt( name = "chaosOffsetAlongBranchRoot", fn=OpenMaya.MRampAttribute, shortName="orrt", type="Ramp" )
	lightningBoltNode.hlp.createAtt( name = "numChildrenRootOverride", fn=numAttr, shortName="ncro", type=OpenMaya.MFnNumericData.kBoolean, default=0 )	
	lightningBoltNode.hlp.createAtt( name = "transfertNumChildrenRoot", fn=numAttr, shortName="tncrt", type=OpenMaya.MFnNumericData.kDouble, default=1.0, exceptAffectList=['samplingDummyOut'] )	
	lightningBoltNode.hlp.createAtt( name = "transfertNumChildrenRandRoot", fn=numAttr, shortName="tncrrt", type=OpenMaya.MFnNumericData.kDouble, default=1.0, exceptAffectList=['samplingDummyOut'] )	



	# APV Special overrides
	lightningBoltNode.hlp.createAtt( name = "elevationRootOverride", fn=numAttr, shortName="ero", type=OpenMaya.MFnNumericData.kBoolean, default=0 )	
	lightningBoltNode.hlp.createAtt( name = "elevationAlongBranchRoot", fn=OpenMaya.MRampAttribute, shortName="errt", type="Ramp" )
	lightningBoltNode.hlp.createAtt( name = "elevationRandAlongBranchRoot", fn=OpenMaya.MRampAttribute, shortName="errrt", type="Ramp" )

	# APV Branches overrides
	lightningBoltNode.hlp.createAtt( name = "radiusRootOverride", fn=numAttr, shortName="rro", type=OpenMaya.MFnNumericData.kBoolean, default=0 )	
	lightningBoltNode.hlp.createAtt( name = "radiusAlongBranchRoot", fn=OpenMaya.MRampAttribute, shortName="rrrt", type="Ramp" )
	lightningBoltNode.hlp.createAtt( name = "transfertRadiusRoot", fn=numAttr, shortName="trr", type=OpenMaya.MFnNumericData.kDouble, default=1.0, exceptAffectList=['samplingDummyOut'] )

	lightningBoltNode.hlp.createAtt( name = "lengthRootOverride", fn=numAttr, shortName="clro", type=OpenMaya.MFnNumericData.kBoolean, default=0 )	
	lightningBoltNode.hlp.createAtt( name = "childLengthAlongPathRoot", fn=OpenMaya.MRampAttribute, shortName="clrrt", type="Ramp" )
	lightningBoltNode.hlp.createAtt( name = "transfertLengthRoot", fn=numAttr, shortName="tclr", type=OpenMaya.MFnNumericData.kDouble, default=1.0, exceptAffectList=['samplingDummyOut'] )

	lightningBoltNode.hlp.createAtt( name = "intensityRootOverride", fn=numAttr, shortName="iro", type=OpenMaya.MFnNumericData.kBoolean, default=0 )	
	lightningBoltNode.hlp.createAtt( name = "intensityAlongPathRoot", fn=OpenMaya.MRampAttribute, shortName="irrt", type="Ramp" )
	lightningBoltNode.hlp.createAtt( name = "transfertIntensityRoot", fn=numAttr, shortName="tir", type=OpenMaya.MFnNumericData.kDouble, default=1.0, exceptAffectList=['samplingDummyOut'] )

# OUTPUTS
	lightningBoltNode.hlp.createAtt( name = "samplingDummyOut", isInput=False, fn=numAttr, shortName="sdo", type=OpenMaya.MFnNumericData.kInt)
	lightningBoltNode.hlp.createAtt( name = "outputMesh", isInput=False, fn=typedAttr, shortName="out", type=OpenMaya.MFnData.kMesh)	
	lightningBoltNode.hlp.addAllAttrs()
	lightningBoltNode.hlp.generateAffects()

def cleanupClass():
	delattr(lightningBoltNode,'defaultDetailValue')
	delattr(lightningBoltNode,'defaultMaxGeneration')
	delattr(lightningBoltNode,'defaultNumChildren')
	delattr(lightningBoltNode,'defaultNumChildrenRand')
	delattr(lightningBoltNode,'lightningModule')
	delattr(lightningBoltNode,'LPM')
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
