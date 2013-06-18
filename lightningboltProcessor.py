#!/usr/bin/python
# -*- coding: iso-8859-1 -*-

import numpy as np
import math
import random
import sys

class LBP_msCommandException(Exception):
    def __init__(self,message):
        self.message = '[LBP] '+message
    
    def __str__(self):
        return self.message

def arndSphereVect(N):
    """Generate N random points on a unit sphere, equally spaced on the
    surface of the sphere, and return them as columns of an array.
    """   
    sph = np.empty( (N,3), np.float )
    
    sph[:,2] = np.random.uniform(-1.0,1.0,N) # z-coordinates
    z2 = np.sqrt(1.0 - sph[:,2]**2)
    phi = (2.0 * math.pi) * np.random.random( N )
    sph[:,0] = z2 * np.cos(phi) # x 
    sph[:,1] = z2 * np.sin(phi) # y
    return sph

class lightningBoltProcessor:

	kAPV_Radius = 0
	kAPV_Intensity = 1
	kAPV_Offset = 2
	kAPV_MaxAttributes = 3

	kV_numChildrens = 0
	kV_randNumChildrens = 1
	kV_MaxAttributes = 2


	def __init__( self, outSoftwareFunc = None ):
		self.outSoftwareFunc = outSoftwareFunc

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

		# has no variation along the path
		# O numchildens (nombre d'enfant genere obligatoirement)
		# 1 randNumChildrens (nombre d'enfant potentiellement genere aleatoirement en plus)

		self.VInputsGeneral = np.zeros((lightningBoltProcessor.kV_MaxAttributes),np.float32)

		#for testing
		self.VInputsGeneral[0] = 2
		self.VInputsGeneral[1] = 2

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

	def setVGeneral(self, num, value):
		self.VInputsGeneral[num] = value
	
	def initializeProcessor( self ):
		self.startSeedPoints = []
		self.numStartSeedBranch = 0
		self.startSeedBranchesAttributes = []

	def addSeedPointList( self, pointList):
		self.startSeedPoints.extend( pointList )
		#self.startSeedPoints.append([0,0,0]) # add an extra blank point
		self.numStartSeedBranch =  self.numStartSeedBranch + 1
		self.startSeedBranchesAttributes.append(self.APVInputsMultiplier1.reshape(1,-1))

		# for testing
		self.startSeedPoints.extend( [[ point[0],point[1]+2,point[2] ] for point in pointList] )
		#self.startSeedPoints.append([0,0,0]) # add an extra blank point
		self.numStartSeedBranch =  self.numStartSeedBranch + 1
		self.startSeedBranchesAttributes.append(self.APVInputsMultiplier1.reshape(1,-1)*0.5)
	'''
	def generate( self, branchInputs, APV, doGenerateChilds ):
		seedPath, APVMult = branchInputs
		sys.stderr.write('APVMult '+str(APVMult)+'\n')

		APArray = APV * APVMult # the attribute path Array contain every attribute value at every point

		offsetArray = np.array( arndSphereVect(self.maxPoints) ,np.float32)
		#import sys
		#sys.stderr.write('offetArray (Points)'+str(offsetArray)+'\n')
		#sys.stderr.write('offset '+str(APArray[:,lightningBoltProcessor.kAPV_Offset].reshape(self.maxPoints,-1))+'\n')
		seedPath += offsetArray* APArray[:,lightningBoltProcessor.kAPV_Offset].reshape(self.maxPoints,-1)
		#sys.stderr.write('seedPath '+str(seedPath)+'\n')

		# decal d'un point l'array seedPath, soustrait seedPath, on obtient un vecteur, on discard le dernier element multiplie par lui meme, on fit la somme des composantes et la racine carree -> norme
		#vectors = (np.roll(seedPath,-3) - seedPath)[:-1]
		#norms = np.sqrt( (vectors**2).sum(-1) )
		
		#np.divide(vectors,norms[:,np.newaxis],vectors)
		#sys.stderr.write('vectors '+str(vectors)+'\n')
		
		#np.sqrt((vectors ** 2).sum(-1))[..., np.newaxis]
		if doGenerateChilds:
			numChilds = self.VInputsGeneral[0] + random.randint(0, self.VInputsGeneral[1])
			epsilon = 0.000001
			childsParams = np.random.uniform(epsilon, float(self.maxPoints-1) - epsilon, numChilds)
			for c in childsParams:
				sys.stderr.write('child param '+str(c)+'\n')
				id0 = int(c)
				id1 = id0 + 1
				t = c - float(id0)
				seedAPVal = (1-t)*APArray[id0] + t*APArray[id1]
				sys.stderr.write('seedAPVal '+str(seedAPVal)+'\n')


		return (seedPath, APArray) , []
	'''

	def generate( self, batch, APV, isLooping, doGenerateChilds ):
		
		batchSize, seedPath, APVMults = batch
		sys.stderr.write('APVMults '+str(APVMults)+'\n')
		APVs = np.tile(APV,(batchSize,1,1)) 
		sys.stderr.write('APVs '+str(APVs)+'\n')
		APArray = APVs * APVMults # the attribute path Array contain every attribute value at every point
		sys.stderr.write('APArray '+str(APArray)+'\n')

		offsetArray = (np.array( arndSphereVect(batchSize*self.maxPoints) ,np.float32)).reshape(batchSize,self.maxPoints,3)
		#import sys
		#sys.stderr.write('offetArray (Points)'+str(offsetArray)+'\n')
		#sys.stderr.write('offset '+str(APArray[:,lightningBoltProcessor.kAPV_Offset].reshape(self.maxPoints,-1))+'\n')
		sys.stderr.write('seedPath '+str(seedPath)+'\n')
		seedPathPointsView = seedPath.reshape(batchSize,self.maxPoints,3) #[:,:-1] #just to remove
		sys.stderr.write('seedPathPointsView '+str(seedPathPointsView)+'\n')
		sys.stderr.write('offsetArray '+str(offsetArray)+'\n')
		#temp = APArray[:,:,:,lightningBoltProcessor.kAPV_Offset].reshape(1,1,batchSize,self.maxPoints,-1)
		#sys.stderr.write('temp '+str(temp)+'\n')
		#sys.stderr.write('offsetArray*temp '+str((offsetArray*temp)[0,0])+'\n')

		seedPathPointsView += (offsetArray*APArray[:,:,:,lightningBoltProcessor.kAPV_Offset].reshape(1,1,batchSize,self.maxPoints,-1))[0,0]
		sys.stderr.write('seedPathPointsView '+str(seedPathPointsView)+'\n')
		sys.stderr.write('seedPath '+str(seedPath)+'\n')

		# let's create the frame on all the paths
		frameTemplate = np.array( [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]], np.float32 )
		frames = np.tile(frameTemplate,(batchSize,self.maxPoints,1,1))

		# -------------- calculate the vector front at every point
		frameFronts = frames[:,:,0,:-1]
		#sys.stderr.write('frameFronts '+str(frameFronts)+'\n')
		#seedPathView = seedPath.reshape(batchSize,self.maxPoints+1,-1)

		#sys.stderr.write('seedPathView '+str(seedPathView)+'\n')
		unitDir = (np.roll( seedPathPointsView, -1, axis=1 )[:,:-1])
		sys.stderr.write('unitDir '+str(unitDir)+'\n')
		unitDir -= seedPathPointsView[:,:-1]
		sys.stderr.write('unitDir '+str(unitDir)+'\n')
		normFrontSqr = np.sqrt(((unitDir**2).sum(-1)).reshape(batchSize,self.maxPoints-1,1))
		sys.stderr.write('normFrontSqr '+str(normFrontSqr)+'\n')
		unitDir /= normFrontSqr
		sys.stderr.write('unitDir '+str(unitDir)+'\n')

		sys.stderr.write('frameFronts '+str(frameFronts)+'\n')
		frameFronts[:,:-1,:][...] = unitDir
		frameFronts[:,1:,:] += unitDir
		sys.stderr.write('frameFronts '+str(frameFronts)+'\n')

		if isLooping :
			frameFronts[:,0,:] += frameFronts[:,-1,:] # we add the last direction of the branch to the first direction
			# we normalize all the direction except the last one ( -> :-1 )
			normFrontSqr = np.sqrt(((frameFronts[:,:-1,:]**2).sum(-1)).reshape(batchSize,self.maxPoints-1,1))
			frameFronts[:,:-1,:] /= normFrontSqr
		else : # we normalize all the direction except the first one and the last one ( -> 1:-1 )
			normFrontSqr = np.sqrt(((frameFronts[:,1:-1,:]**2).sum(-1)).reshape(batchSize,self.maxPoints-2,1))
			frameFronts[:,1:-1,:] /= normFrontSqr

		sys.stderr.write('frameFronts '+str(frameFronts)+'\n')

	#------ the algorithm to setup the frame for each point need the first frame to be completely initialized for every branch
		firstFrameViewFront = frames[:,0,0,:-1]
		firstFrameViewUp = frames[:,0,1,:-1]
		firstFrameViewSide = frames[:,0,2,:-1]
		#sys.stderr.write('firstFrameViewUp '+str(firstFrameViewUp)+'\n')

		# set the Up vector 
		firstFrameViewUp[:] = [0,1,0] # we initialize Up with vector Y
		#sys.stderr.write('firstFrameViewUp '+str(firstFrameViewUp)+'\n')
		verticalStartBranch = np.where( np.abs( (firstFrameViewUp*firstFrameViewFront).sum(-1)) > 0.99999 )
		firstFrameViewUp[verticalStartBranch] = [1,0,0] # where it's vertical let's start with this vector
		#sys.stderr.write('firstFrameViewUp '+str(firstFrameViewUp)+'\n')
		# with a valid Up we can deduce the side then the up again to have a proper frame
		firstFrameViewSide[:] = np.cross( firstFrameViewFront, firstFrameViewUp )
		#sys.stderr.write('normSqrt '+str( ((firstFrameViewSide**2).sum(-1)).reshape(-1,1) )+'\n')
		firstFrameViewSide /= np.sqrt( ((firstFrameViewSide**2).sum(-1)).reshape(-1,1) )

		firstFrameViewUp[:] = np.cross( firstFrameViewSide, firstFrameViewFront )
		#sys.stderr.write('frames[:,0,0,:-1] '+str(frames[:,0,:,:-1])+'\n')

		sys.stderr.write('frames '+str(frames)+'\n')		
	#------  now we can calculate the other frame on the path by iterating from the start point
		for i in range(1,self.maxPoints):
			frontPrevPoint = frames[:,i-1,0,:-1]
			upPrevPoint = frames[:,i-1,1,:-1]

			frontCurrentPoint = frames[:,i,0,:-1]
			upCurrentPoint = frames[:,i,1,:-1]
			sideCurrentPoint = frames[:,i,2,:-1]

			sys.stderr.write('unitDir '+str(unitDir[:,i-1,:])+'\n')		
			sys.stderr.write('upPrevPoint '+str(upPrevPoint)+'\n')		
			upL = ((unitDir[:,i-1,:]*upPrevPoint).sum(-1)*-2.0).reshape(-1,1)*unitDir[:,i-1,:]
			upL += upPrevPoint
			sys.stderr.write('upL '+str(upL)+'\n')
			frL = ((unitDir[:,i-1,:]*frontPrevPoint).sum(-1)*2.0).reshape(-1,1)*unitDir[:,i-1,:]
			frL -= frontPrevPoint
			v2 = frL # make v2 the same object as frL (frL is used only for v2)
			v2 += frontCurrentPoint
			c2 = (v2*v2).sum(-1).reshape(-1,1) # dot
			sys.stderr.write('c2 '+str(c2)+'\n')

			upCurrentPoint[...] = ((v2*upL).sum(-1)*-2.0).reshape(-1,1)*v2
			upCurrentPoint /= c2
			upCurrentPoint += upL

			sideCurrentPoint[...] = np.cross(frontCurrentPoint,upCurrentPoint)

		sys.stderr.write('frames[:,:,:-1,:-1] '+str(frames[:,:,:-1,:-1])+'\n')
		sys.stderr.write('APArray[0,:,:,lightningBoltProcessor.kAPV_Radius,np.newaxis,np.newaxis] '+str(APArray[0,:,:,lightningBoltProcessor.kAPV_Radius,np.newaxis,np.newaxis])+'\n')
		# multiply frames by corresponding Radius
		frames[:,:,:-1,:-1] *= APArray[0,:,:,lightningBoltProcessor.kAPV_Radius,np.newaxis,np.newaxis]

		# load points in frames
		frames[:,:,3,:-1] = seedPathPointsView
		sys.stderr.write('frames '+str(frames)+'\n')

		childBatches = None
		if doGenerateChilds:
			numChilds = self.VInputsGeneral[0] + random.randint(0, self.VInputsGeneral[1])
			epsilon = 0.000001
			childsParams = np.random.uniform(epsilon, float(self.maxPoints-1) - epsilon, numChilds)
			for c in childsParams:
				sys.stderr.write('child param '+str(c)+'\n')
				id0 = int(c)
				id1 = id0 + 1
				t = c - float(id0)
				seedAPVal = (1-t)*APArray[0,0,id0] + t*APArray[0,0,id1]
				sys.stderr.write('seedAPVal '+str(seedAPVal)+'\n')


		return frames, childBatches

	def process(self ):

		maxGeneration = 3
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

		sys.stderr.write('startSeedBranchesAttributes '+str(self.startSeedBranchesAttributes)+'\n')
		startSeedAttributes = np.array( [self.startSeedBranchesAttributes] ) 
		sys.stderr.write('startSeedAttributes '+str(startSeedAttributes)+'\n')
		APV = firstGenerationAPV

		# The result returned will be points in the form of a huge list of coordinate, there will be 2 lists to separate ring and non ring lightning
		resultLoopingFrames = []
		numLoopingLightningBranch = 0
		resultFrames = []
		numLightningBranch = 0
		isLooping = False

		batchSize = self.numStartSeedBranch
		batch = batchSize, startSeedPath, startSeedAttributes
		while batch is not None:
			outFrames, childBatch = self.generate( batch, APV, isLooping, False) #currentGeneration<maxGeneration

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

		# here deduce points from frame and circles and add them to resultPoints
		# transform a circle of point for each frame
		circles = np.tile(pointsOfCircle,(len(resultFrames[0]),1,1)) # make enough circles

		resultPoints = [ coord for array in resultFrames for k in range(len(array)) for point in (np.dot(circles[k],array[k])).tolist() for coord in point  ] 
		resultLoopingPoints = [] # TODO
		numLoopingLightningBranch = 0 # TODO

		return self.outSoftwareFunc( resultLoopingPoints, numLoopingLightningBranch, resultPoints, numLightningBranch, self.maxPoints, tubeSide )

