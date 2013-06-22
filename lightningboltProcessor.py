#!/usr/bin/python
# -*- coding: iso-8859-1 -*-

'''
================================================================================
* VERSION 0.0
================================================================================
* AUTHOR:
Mathieu Sauvage mathieu@hiddenforest.fr
================================================================================
* INTERNET SOURCE:
================================================================================
* DESCRIPTION:
This is a lightning Processor designed to process batch of lightnings and let a
good control over the shape of the lightnings.  So it uses numpy to try to have
a python code as efficient as possible
================================================================================
* USAGE:
================================================================================
* TODO:
- putting the length
- implementing the transfert of attribute
- generate full rotation matrix instead of just vector on sphere ( so we can transmit the frame and we don't need to calculate the start Up for every branch or maybe not)
- good noise
- clean the inputs that are hardcoded like tubeSide
- more parameters, seed controls, etc...
================================================================================
'''


import numpy as np
import math
import random
import sys
import time

class LBP_msCommandException(Exception):
    def __init__(self,message):
        self.message = '[LBP] '+message
    
    def __str__(self):
        return self.message

class avgTimer:
	def __init__( self, numValues ):
		self.arrayOfTime = [0.0]*numValues
		self.nextIndex = 0
		self.startTime = 0.0

	def average(self):
		return sum(self.arrayOfTime)/float(len(self.arrayOfTime))

	def start( self ):
		self.startTime = time.clock()

	def stop( self ):
		self.arrayOfTime[self.nextIndex] = time.clock() - self.startTime
		self.nextIndex = (self.nextIndex + 1 ) % len(self.arrayOfTime)
		return self.average()

def arndSphereVect(N):
    """Generate N random points on a unit sphere, equally spaced on the
    surface of the sphere, and return them as columns of an array.
    """   
    sph = np.empty( (N,3), np.float32 )
    
    sph[:,2] = np.random.uniform(-1.0,1.0,N) # z-coordinates
    z2 = np.sqrt(1.0 - sph[:,2]**2)
    phi = (2.0 * math.pi) * np.random.random( N )
    sph[:,0] = z2 * np.cos(phi) # x 
    sph[:,1] = z2 * np.sin(phi) # y
    return sph

# TODO function to return an array of elevation Vectors
'''
def arndSphereElevation( N, elevation, amplitude ):
	phi = random.uniform(0,2.0*math.pi)
	costheta = 1.0-2.0*elevation + random.uniform(-amplitude,amplitude) #random.random(-1,1)
	if costheta<-1.0 : costheta = -1.0
	if costheta>1.0 : costheta = 1.0	
	theta = math.acos(costheta)

	return OpenMaya.MVector( math.sin( theta) * math.cos( phi ) , math.sin( theta) * math.sin( phi ), math.cos( theta ) )
'''

def generateRandomParams( seeds ):
	pass

class nRand():
	def __init__(self, seed=0):
		self.rndState = None
		self.nprndState = None
		self.seed(int(seed))

	def seed( self, seedValue):
		random.seed(seedValue)
		self.rndState = random.getstate()

		self.nprndState = np.random.RandomState(seedValue)
		#np.random.seed(seedValue)
		#self.nprndState = np.random.get_state()

	def resume( self ):
		random.setstate(self.rndState)
		#np.random.set_state(self.nprndState)

	def randInt( self, min, max ):
		return random.randint(min, max)

	def npaRand( self, min, max, sizeArray ):
		return self.nprndState.uniform(min,max,size=sizeArray)

	def npaRandEpsilon( self, min, max, N, epsilon=0.00001):
		return self.nprndState.uniform(min+epsilon,max-epsilon,N)

	def npaRandSphereElevation( self, aElevation, aAmplitude, N ):
		sphPoints = np.empty( (N,3), np.float32 )
		phi = self.nprndState.uniform(0.0,2.0*np.pi,N)
		theta = np.arccos( np.clip( aAmplitude*self.nprndState.uniform( -1.0, 1.0, N ) -2.0*aElevation + 1.0, -1.0,1.0 ))
		sphPoints[:,0] = sphPoints[:,1] = np.sin( theta )
		sphPoints[:,2] =  np.cos( theta )
		sphPoints[:,0] *= np.cos( phi )
		sphPoints[:,1] *= np.sin( phi )
		return sphPoints

def npaRandSphereElevation( aElevation, aAmplitude, aPhi, aAmplitudeRand,  N ):
	sphPoints = np.empty( (N,3), np.float32 )
	#sys.stderr.write( 'aAmplitude '+str(aAmplitude)+'\n' )
	#sys.stderr.write( 'aAmplitudeRand '+str(aAmplitudeRand)+'\n' )
	sys.stderr.write( 'mult '+str(aAmplitude*aAmplitudeRand)+'\n' )
	theta = np.arccos( np.clip( aAmplitude*aAmplitudeRand - 2.0*aElevation + 1.0, -1.0,1.0 ))
	sphPoints[:,1] = sphPoints[:,2] = np.sin( theta )
	sphPoints[:,0] =  np.cos( theta )
	sphPoints[:,1] *= np.cos( aPhi )
	sphPoints[:,2] *= np.sin( aPhi )
	return sphPoints


class lightningBoltProcessor:

	# Along Path values Enum (attributes transmitted to children by mult)
	kAPV_Radius = 0
	kAPV_Intensity = 1
	kAPV_Offset = 2
	kAPV_Elevation = 3
	kAPV_ElevationRand = 4
	kAPV_MaxAttributes = 5

	# Values enum (attributes transmitted to children by mult)
	kV_numChildrens = 0
	kV_randNumChildrens = 1
	kV_branchingTimeMult = 2
	kV_MaxAttributes = 3

	# Generic Values enum (no transmitted by function)
	kGV_seedBranching = 0
	kGV_MaxAttributes = 1
	

	def __init__( self, outSoftwareFunc = None ):
		self.outSoftwareFunc = outSoftwareFunc

		self.timer = avgTimer(30)

		self.detail = -1
		self.maxPoints = 0

		# APVariations are Along Path Variations (sampled curves), APVariations are the variations that are given to the system as inputs (they are maybe set by external ramps)
		# APVariations is a big numpy array where lines are the parameter along the path and column are alls the attributes
		# column 0 is Radius
		# column 1 is Intensity
		# column 2 is Offset
		self.APVInputs1 = None
		# this is a vector with all the multiplicator to apply to APVInputs
		self.APVInputsMultiplier1 = np.zeros((lightningBoltProcessor.kAPV_MaxAttributes),np.float32)

		# has no variation along the path is transmitted to child
		# O numchildens (nombre d'enfant genere obligatoirement)
		# 1 randNumChildrens (nombre d'enfant potentiellement genere aleatoirement en plus)
		# 2 branchingTimeMult (Multiplicator to timeBranching used to generate the childs number and params)
		self.VInputs = np.zeros((lightningBoltProcessor.kV_MaxAttributes),np.float32)

		# generic values (is not transmitted by function to child)
		# 0 seedBranching (the seed that decided the number of childrens and the parameters of each one)
		self.GVInputs = np.zeros((lightningBoltProcessor.kGV_MaxAttributes),np.float32)

		#for testing
		self.VInputs[lightningBoltProcessor.kV_numChildrens] = 2
		self.VInputs[lightningBoltProcessor.kV_randNumChildrens] = 2
		self.GVInputs[lightningBoltProcessor.kGV_seedBranching] = 437

	def setDetail( self, detail):
		if detail == self.detail:
			return
		self.detail = detail
		self.maxPoints = 2 + 2**detail
		self.APVInputs1 = np.zeros((self.maxPoints,lightningBoltProcessor.kAPV_MaxAttributes),np.float32)
	
	def getAPVariation1(self, num):
		''' to get the array specific to one attribute '''
		return self.APVInputs1[:,num]

	def setAPVMult1(self, num, value):
		self.APVInputsMultiplier1[num] = value

	def setValue(self, num, value):
		self.VInputs[num] = value

	def setGenericValue(self, num, value):
		self.GVInputs[num] = value
	
	def initializeProcessor( self ):
		self.startSeedPoints = []
		self.numStartSeedBranch = 0
		self.startSeedBranchesAPVMult = []
		self.startSeedBranchesV = []
		self.startSeedBranchesGV = []

	def addSeedPointList( self, pointList):
		self.startSeedPoints.extend( pointList )
		#self.startSeedPoints.append([0,0,0]) # add an extra blank point
		self.numStartSeedBranch =  self.numStartSeedBranch + 1
		self.startSeedBranchesAPVMult.append(self.APVInputsMultiplier1.reshape(1,-1))
		self.startSeedBranchesV.append(self.VInputs)
		self.startSeedBranchesGV.append(self.GVInputs)

		# for testing
		self.startSeedPoints.extend( [[ point[0],point[1]+2,point[2] ] for point in pointList] )
		#self.startSeedPoints.append([0,0,0]) # add an extra blank point
		self.numStartSeedBranch =  self.numStartSeedBranch + 1
		self.startSeedBranchesAPVMult.append(self.APVInputsMultiplier1.reshape(1,-1)*0.5)
		self.startSeedBranchesV.append(self.VInputs)
		self.startSeedBranchesGV.append(self.GVInputs)

		self.startSeedPoints.extend( [[ point[0],point[1]+4,point[2] ] for point in pointList] )
		#self.startSeedPoints.append([0,0,0]) # add an extra blank point
		self.numStartSeedBranch =  self.numStartSeedBranch + 1
		self.startSeedBranchesAPVMult.append(self.APVInputsMultiplier1.reshape(1,-1)*2.0)
		self.startSeedBranchesV.append(self.VInputs)
		self.startSeedBranchesGV.append(self.GVInputs)

#sys.stderr.write('seedPathPointsView '+str(seedPathPointsView)+'\n')
	def generate( self, batch, APV, branchingTime, isLooping, doGenerateChilds ):
		# unpack
		batchSize, seedPath, APVMults, Values, GValues = batch

		#sys.stderr.write('APV '+str(APV)+'\n')
		#sys.stderr.write('batchSize '+str(batchSize)+'\n')
		#sys.stderr.write('seedPath '+str(seedPath)+'\n')
		#sys.stderr.write('APVMults '+str(APVMults)+'\n')
		#sys.stderr.write('Values '+str(Values)+'\n')
		totalBatchPoints = len(seedPath) # this should be self.maxPoints*batchSize

		# duplicate the Path variation values to have one for each branch
		APVs = np.tile(APV,(batchSize,1,1))

		# get the final values at each path point ( the value from the ramp multiply by the multiplicator )
		APArray = APVs * APVMults # the attribute path Array contain every attribute value at every point
		#sys.stderr.write('APArray '+str(APArray)+'\n')

		# get the a random vector for every path point (that will offset the position of seedPath)
		offsetArray = (arndSphereVect(totalBatchPoints)).reshape(batchSize,self.maxPoints,3)
		#sys.stderr.write('offsetArray '+str(offsetArray)+'\n')
		
		seedPathPointsView = seedPath.reshape(batchSize,self.maxPoints,3)

		# multiply all the random vector by the offset multiplicator at each point
		#test = APArray[:,:,:,lightningBoltProcessor.kAPV_Offset].reshape(1,1,batchSize,self.maxPoints,-1)
		#sys.stderr.write('test '+str(test)+'\n')

		seedPathPointsView += (offsetArray*APArray[:,:,:,lightningBoltProcessor.kAPV_Offset].reshape(1,1,batchSize,self.maxPoints,-1))[0,0]
		#sys.stderr.write('seedPath '+str(seedPath)+'\n')

	# from here we compute the frame ( unit front, unit up, unit side ) at each point
	# the final frame is a full 4,4 matrix, like this :
	#  fx ux sx px 
	#  fy uy sy py
	#  fz uz sz pz
	#  0  0  0  1
	#  where front is (fx,fy,fz,0)
	#  where side is (sx,sy,sz,0)
	#  where up is (ux,uy,uz,0)
	#  where path point is (px,py,pz,1)
	# this way we have a full transformation matrix with a good translation that we can use to get all the tube points
	# algorithm is this:
	# - compute vector between all the points Di
	# - normalize Di
	# - unit front = Di-1 + Di (at each point the frame is "tangent")
	# - normalize unit front and unit front is good
	# - complete for every branch the initial frame. Get a unit side from an arbitrary cross(front,[0,1,0]) or cross(front, [1,0,0]) if [0,1,0] and front are parallel. Then get the unit Up by cross( side, front)
	# - then apply the algorithm (Double Reflection) for each branch ( http://research.microsoft.com/en-us/um/people/yangliu/publication/computation%20of%20rotation%20minimizing%20frames.pdf ) . This algorithm compute a good frame from a previous frame, so we iteratively compute all the frames along the paths. 

		# let's create the frame for every points on every branch
		frameTemplate = np.array( [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]], np.float32 )
		frames = np.tile(frameTemplate,(batchSize,self.maxPoints,1,1))

		frameFronts = frames[:,:,:-1,0] # get a view for the front only
		
		# compute vector between all the points Di = roll( pathPoint, -1 ) - pathPoint
		dirVectors = np.roll( seedPathPointsView, -1, axis=1 )
		unitDir = dirVectors[:,:-1]
		unitDir -= seedPathPointsView[:,:-1]
		normFrontSqr = np.sqrt(((unitDir**2).sum(-1)).reshape(batchSize,self.maxPoints-1,1))
		unitDir /= normFrontSqr

		# front = Di-1 + Di
		frameFronts[:,:-1,:][...] = unitDir
		frameFronts[:,1:,:] += unitDir

		# normalize front (different case if we loop (the first point of the branch is different when we loop) )
		if isLooping :
			frameFronts[:,0,:] += frameFronts[:,-1,:] # we add the last direction of the branch to the first direction
			# we normalize all the direction except the last one ( -> :-1 )
			normFrontSqr = np.sqrt(((frameFronts[:,:-1,:]**2).sum(-1)).reshape(batchSize,self.maxPoints-1,1))
			frameFronts[:,:-1,:] /= normFrontSqr
		else : # we normalize all the direction except the first one and the last one ( -> 1:-1 )
			normFrontSqr = np.sqrt(((frameFronts[:,1:-1,:]**2).sum(-1)).reshape(batchSize,self.maxPoints-2,1))
			frameFronts[:,1:-1,:] /= normFrontSqr

		#complete for every branch the initial frame
		firstFrameViewFront = frames[:,0,:-1,0]
		firstFrameViewUp = frames[:,0,:-1,1]
		firstFrameViewSide = frames[:,0,:-1,2]

		firstFrameViewUp[:] = [0,1,0] # we initialize Up with vector Y
		firstFrameViewFrontY = frames[:,0,1,0]
		#sys.stderr.write('orig where '+str(np.where( np.abs( (firstFrameViewUp*firstFrameViewFront).sum(-1)) > 0.99999 ))+'\n')
		#sys.stderr.write('new where '+str( np.where( np.abs(firstFrameViewFrontY)>0.99999 )  )+'\n')
		verticalStartBranch = np.where( np.abs(firstFrameViewFrontY)>0.99999 )
		firstFrameViewUp[verticalStartBranch] = [1,0,0] # where it's vertical let's start with this vector instead

		# with a valid Up we can deduce the side then the up again to have a proper frame
		firstFrameViewSide[:] = np.cross( firstFrameViewFront, firstFrameViewUp )
		firstFrameViewSide /= np.sqrt( ((firstFrameViewSide**2).sum(-1)).reshape(-1,1) )

		firstFrameViewUp[:] = np.cross( firstFrameViewSide, firstFrameViewFront )

		# now apply the algorithm (Double Reflection) for each branch
		for i in range(1,self.maxPoints):
			frontPrevPoint = frames[:,i-1,:-1,0]
			upPrevPoint = frames[:,i-1,:-1,1]

			frontCurrentPoint = frames[:,i,:-1,0]
			upCurrentPoint = frames[:,i,:-1,1]
			sideCurrentPoint = frames[:,i,:-1,2]

			upL = ((unitDir[:,i-1,:]*upPrevPoint).sum(-1)*-2.0).reshape(-1,1)*unitDir[:,i-1,:]
			upL += upPrevPoint
			frL = ((unitDir[:,i-1,:]*frontPrevPoint).sum(-1)*2.0).reshape(-1,1)*unitDir[:,i-1,:]
			frL -= frontPrevPoint
			v2 = frL # make v2 the same object as frL (frL is used only for v2)
			v2 += frontCurrentPoint
			c2 = (v2*v2).sum(-1).reshape(-1,1) # dot

			upCurrentPoint[...] = ((v2*upL).sum(-1)*-2.0).reshape(-1,1)*v2
			upCurrentPoint /= c2
			upCurrentPoint += upL
			sideCurrentPoint[...] = np.cross(frontCurrentPoint,upCurrentPoint)

		# multiply frames by corresponding Radius ( we scale only the vectors )
		frames[:,:,:-1,:-1] *= APArray[0,:,:,lightningBoltProcessor.kAPV_Radius,np.newaxis,np.newaxis]

		# load branch points in frames to complete the transformation matrix (we are done with the frames!)
		frames[:,:,:-1,3] = seedPathPointsView

	# now we can generate childs if we need
		childsBatch = None
		if doGenerateChilds:
			#sys.stderr.write('GValues[:,lightningBoltProcessor.kGV_seedBranching] '+str(GValues[:,lightningBoltProcessor.kGV_seedBranching])+'\n')
			GValues[:,lightningBoltProcessor.kGV_seedBranching] += np.floor( Values[:,lightningBoltProcessor.kV_branchingTimeMult]*branchingTime )

			allChildsParams = []
			allChildsBranchParentId = ([],[])
			allRandValuesForSpherePt = []
			# we have to loop through the branches, no choice to generate the randoms values with a seed for each branch
			for i in range(batchSize): 
				randBranch = nRand(GValues[i,lightningBoltProcessor.kGV_seedBranching])
				Values[i,lightningBoltProcessor.kV_numChildrens] += randBranch.randInt(0, Values[i,lightningBoltProcessor.kV_randNumChildrens] )
				allChildsParams.append( randBranch.npaRandEpsilon(float(i*self.maxPoints), float((i+1)*self.maxPoints-1), Values[i,lightningBoltProcessor.kV_numChildrens] ) )
				# random values used to generate random direction from parent branch
				allRandValuesForSpherePt.append( randBranch.npaRand(0.0,2.0,(2,Values[i,lightningBoltProcessor.kV_numChildrens]) ) )
				allChildsBranchParentId[0].extend( [i]*Values[i,lightningBoltProcessor.kV_numChildrens] )

			totalChildNumber = len(allChildsBranchParentId[0])

			APVView = APArray.reshape(-1,lightningBoltProcessor.kAPV_MaxAttributes)
			#sys.stderr.write('APArray '+str(APArray)+'\n')
			blend = np.array(allChildsParams, np.float32).reshape(-1)
			#sys.stderr.write('blend '+str(blend)+'\n')
			indexT0 = np.array(allChildsParams, np.uint32).reshape(-1)
			APVViewT0 = APVView[indexT0]
			#sys.stderr.write('indexT0 '+str(indexT0)+'\n')
			#sys.stderr.write('indexT0+1 '+str(indexT0+1)+'\n')
			blend -= indexT0
			blend = blend.reshape(-1,1)
			#sys.stderr.write('blend '+str(blend)+'\n')
			#sys.stderr.write('APVView[indexT0]'+str(APVView[indexT0])+'\n')
			#sys.stderr.write('blend*APVView[indexT0]'+str(blend*APVView[indexT0])+'\n')
			APVMultChilds = APVViewT0 + blend*( APVView[indexT0+1] - APVViewT0 )
			ValuesChilds = Values[allChildsBranchParentId[0]]
			GValuesChilds = GValues[allChildsBranchParentId[0]]

			#sys.stderr.write('APVMultChilds '+str(APVMultChilds)+'\n')

			#sys.stderr.write('GValuesChilds '+str(GValuesChilds)+'\n')
		
		# now create frames for segments ( based on dirVectors ) so we can control the direction for the child branch
			# multiple childs can be generated on the same segment, we don't want to calculate extras frame system
			#sys.stderr.write('dirVectors'+str(dirVectors)+'\n')
			paramIdToCalc = np.unique(indexT0)
			reverseToChildId = np.searchsorted( paramIdToCalc, indexT0.flat) # this could be removed if numpy.unique has the reverse_index parameter 

			# from the algorithm of Jeppe Revall Frisvad "Building an Orthonormal Basis from a 3D Unit Vector Without Normalization"
			childFrameCalcTemplate = np.array( [[0,0,0],[0,-1,0],[-1,0,0]], np.float32 ) # front, up, side (initial values are the singularity case of the algorithm)
			childFramesCalc = np.tile(childFrameCalcTemplate,(len(paramIdToCalc),1,1))
			childFramesCalc[:,0] = dirVectors.reshape(-1,3)[paramIdToCalc]

			okIds = np.where( childFramesCalc[:,0,2]>-1.0 )
			#sys.stderr.write('okIds'+str(okIds)+'\n')

			okIdsFronts = childFramesCalc[okIds[0],0,:]
			#sys.stderr.write('okIdsFronts '+str(okIdsFronts)+'\n')
			#sys.stderr.write('okIdsFrontsZ '+str(okIdsFronts[:,2])+'\n')
			#sys.stderr.write('OLD '+str(childFramesCalc[okIds[0],0,2])+'\n')
			
			vala = 1.0/(1.0 + okIdsFronts[:,2] )
			valb = -okIdsFronts[:,0]*okIdsFronts[:,1]*vala
			# up
			childFramesCalc[okIds[0],1,0] = 1.0 - vala*(okIdsFronts[:,0]**2)
			childFramesCalc[okIds[0],1,1] = valb
			childFramesCalc[okIds[0],1,2] = -okIdsFronts[:,0]
			# side
			childFramesCalc[okIds[0],2,0] = valb
			childFramesCalc[okIds[0],2,1] = 1.0 - vala*(okIdsFronts[:,1]**2)
			childFramesCalc[okIds[0],2,2] = -okIdsFronts[:,1]

			# here we have the full array for every point
			childFrames = childFramesCalc[reverseToChildId]
		# now get the elevation for each child branch
			aPhiAndRandAmplitude = np.array(allRandValuesForSpherePt)
			aPhi = aPhiAndRandAmplitude[:,0].reshape(-1)
			aAmplitudeRand = aPhiAndRandAmplitude[:,1].reshape(-1)
			aPhi*=np.pi 
			aAmplitudeRand-=1.0 
			randomVectors = npaRandSphereElevation( APVMultChilds[:,lightningBoltProcessor.kAPV_Elevation], APVMultChilds[:,lightningBoltProcessor.kAPV_ElevationRand],aPhi, aAmplitudeRand, aAmplitudeRand.size  )


			#sys.stderr.write('childFrames'+str(childFrames)+'\n')
			#sys.stderr.write('randomVectors'+str(randomVectors)+'\n')
			childFrames*=randomVectors.reshape(-1,3,1) # here we use the random vector coordinate in the frame of each segments to get the direction of 
			#sys.stderr.write('childFrames'+str(childFrames)+'\n')
			childFrames = childFrames.transpose(0,2,1).sum(-1)
			#sys.stderr.write('childFramesF'+str(childFrames)+'\n')
			
			# COULD BE PRECOMPUTED
			seedPathStep = np.linspace(0.0,1.0,self.maxPoints)
			onesPathStep = np.ones(self.maxPoints,np.float32)

			#childSeedPaths = np.tile(seedPathStep,(len(APVMultChilds),1,1))
			#sys.stderr.write('childSeedPaths'+str(childSeedPaths)+'\n')
			#v2[:,np.newaxis]*p2
			#childFrames[:,0,np.newaxis]*seedPathStep[:,np.newaxis]
			seedPathChilds = childFrames[:,np.newaxis]*seedPathStep[:,np.newaxis]
			#sys.stderr.write('seedPathChilds'+str(seedPathChilds) +'\n' )
			seedPathT0 = seedPath[indexT0]
			startPoints = seedPathT0 + blend*( seedPath[indexT0+1] - seedPathT0 )

			#sys.stderr.write('startPoints'+str(startPoints) +'\n' )
			#tiled = np.tile(startPoints.T,(self.maxPoints,1) )
			#sys.stderr.write('startPoints'+str(startPoints) +'\n' )
			#sys.stderr.write('startPoints[:,np.newaxis]*onesPathStep[:,np.newaxis]'+str(startPoints[:,np.newaxis]*onesPathStep[:,np.newaxis]) +'\n' )
			#sys.stderr.write('tiled'+str(tiled.reshape(totalChildNumber,3,-1) ) +'\n' )

			seedPathChilds += startPoints[:,np.newaxis]*onesPathStep[:,np.newaxis]
			#sys.stderr.write('seedPathChilds1'+str(seedPathChilds[1]) +'\n' )

			childsBatch = ( totalChildNumber, seedPathChilds.reshape(-1,3) , APVMultChilds.reshape(1,totalChildNumber,1,-1), ValuesChilds, GValuesChilds )
			#batchSize, seedPath, APVMults, Values, GValues = batch



			#sys.stderr.write('mult '+str(childFrames*)+'\n')

			#sys.stderr.write('childFramesCalc'+str(childFramesCalc)+'\n')

			#b1 = Vec3f(1.0f - n.x*n.x*a, b, -n.x);
    		#b2 = Vec3f(b, 1.0f - n.y*n.y*a, -n.y);


			#segmentsFrame = np.array( size=(len(allChildsBranchParentId[0]), ) )
			



			# depending on indexT0
			# we have the direction unitDir

			'''
			seeds = np.random.randomInt( 1000, batchSize )

			numChilds = Values[0] + random.randint(0, Values[1])
			epsilon = 0.000001
			childsParams = np.random.uniform(epsilon, float(self.maxPoints-1) - epsilon, numChilds)
			for c in childsParams:
				sys.stderr.write('child param '+str(c)+'\n')
				id0 = int(c)
				id1 = id0 + 1
				t = c - float(id0)
				seedAPVal = (1-t)*APArray[0,0,id0] + t*APArray[0,0,id1]
				sys.stderr.write('seedAPVal '+str(seedAPVal)+'\n')
			'''

		return frames, childsBatch

	def process(self, branchingTime ):
		self.timer.start()

		maxGeneration = 1
		currentGeneration = 0

		firstGenerationAPV  = self.APVInputs1
		normalAPV = self.APVInputs1

		# circle template for tube extrusion
		tubeSide = 4
		pointsOfCircle = np.zeros( (4,tubeSide), np.float32 )
		pointsOfCircle[1] = np.sin(np.linspace(1.5*np.pi,-np.pi*.5, tubeSide, endpoint=False))
		pointsOfCircle[2] = np.cos(np.linspace(1.5*np.pi,-np.pi*.5, tubeSide, endpoint=False))
		pointsOfCircle[3] = np.ones(tubeSide)
		pointsOfCircle = pointsOfCircle.T


		# --- create the first datas (seedPath, attributes ) to process 
		startSeedPath = np.array( self.startSeedPoints )

		#sys.stderr.write('startSeedBranchesAttributes '+str(self.startSeedBranchesAttributes)+'\n')
		startSeedAPVMult = np.array( [self.startSeedBranchesAPVMult] ) 
		startSeedValues = np.array( self.startSeedBranchesV )
		startSeedGenericValues = np.array( self.startSeedBranchesGV )
		

		APV = firstGenerationAPV

		# The result returned will be points in the form of a huge list of coordinate, there will be 2 lists to separate ring and non ring lightning
		resultLoopingFrames = []
		numLoopingLightningBranch = 0
		resultFrames = []
		numLightningBranch = 0
		isLooping = False

		batchSize = self.numStartSeedBranch
		batch = batchSize, startSeedPath, startSeedAPVMult, startSeedValues, startSeedGenericValues
		while batch is not None:
			outFrames, childBatch = self.generate( batch, APV, branchingTime, isLooping, currentGeneration<maxGeneration)

			if isLooping :
				resultLoopingFrames.append(outFrames.reshape(-1,4,4))
				numLoopingLightningBranch += batch[0]
			else:
				resultFrames.append(outFrames.reshape(-1,4,4))
				numLightningBranch += batch[0]

			batch = childBatch
			# set the generation to the general values
			APV = normalAPV
			isLooping = False # only the first generation can loop
			currentGeneration = currentGeneration + 1

		# here deduce points from frame and circles and add them to resultPoints
		# transform a circle of point for each frame
		#circles = np.tile(pointsOfCircle,(len(resultFrames[0]),1,1)) # make enough circles
		#sys.stderr.write('resultFrames[0] '+str(resultFrames[0])+'\n')

		# avec v le vecteur et u la matrice
		# np.dot(v,u) donne les coordonnees transformee par u
		#circles = circles.reshape(-1,4)
		#frameV = resultFrames[0].reshape(-1,4,4)
		resultPoints = [ coord for array in resultFrames for coord in ((np.inner(array,pointsOfCircle)).transpose(0,2,1)).reshape(-1).tolist() ]
		
		#resultPoints = [ coord for array in resultFrames for k in range(len(array)) for point in (np.dot(circles[k],array[k])).tolist() for coord in point  ]

		#sys.stderr.write('resultPoints '+str(resultPoints)+'\n')
		#sys.stderr.write('num points '+str(len(resultPoints))+'\n')

		resultLoopingPoints = [] # TODO
		numLoopingLightningBranch = 0 # TODO

		average = self.timer.stop()

		sys.stderr.write('time '+str(average)+'\n')

		return self.outSoftwareFunc( resultLoopingPoints, numLoopingLightningBranch, resultPoints, numLightningBranch, self.maxPoints, tubeSide )

