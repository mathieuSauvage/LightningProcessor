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
- Child system
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

		'''
		# for testing
		self.startSeedPoints.extend( [[ point[0],point[1]+2,point[2] ] for point in pointList] )
		#self.startSeedPoints.append([0,0,0]) # add an extra blank point
		self.numStartSeedBranch =  self.numStartSeedBranch + 1
		self.startSeedBranchesAttributes.append(self.APVInputsMultiplier1.reshape(1,-1)*0.5)

		self.startSeedPoints.extend( [[ point[0],point[1]+4,point[2] ] for point in pointList] )
		#self.startSeedPoints.append([0,0,0]) # add an extra blank point
		self.numStartSeedBranch =  self.numStartSeedBranch + 1
		self.startSeedBranchesAttributes.append(self.APVInputsMultiplier1.reshape(1,-1)*2.0)
		'''

#sys.stderr.write('seedPathPointsView '+str(seedPathPointsView)+'\n')
	def generate( self, batch, APV, isLooping, doGenerateChilds ):
		# unpack
		batchSize, seedPath, APVMults = batch

		# duplicate the Path variation values to have one for each branch
		APVs = np.tile(APV,(batchSize,1,1))

		# get the final values at each path point ( the value from the ramp multiply by the multiplicator )
		APArray = APVs * APVMults # the attribute path Array contain every attribute value at every point

		# get the a random vector for every path point (that will offset the position of seedPath)
		offsetArray = (np.array( arndSphereVect(batchSize*self.maxPoints) ,np.float32)).reshape(batchSize,self.maxPoints,3)

		seedPathPointsView = seedPath.reshape(batchSize,self.maxPoints,3)

		# multiply all the random vector by the offset multiplicator at each point
		seedPathPointsView += (offsetArray*APArray[:,:,:,lightningBoltProcessor.kAPV_Offset].reshape(1,1,batchSize,self.maxPoints,-1))[0,0]

	# from here we compute the frame ( unit front, unit up, unit side ) at each point
	# the final frame is a full 4,4 matrix, like this :
	# 	unit front  0
	#	unit up     0
	#	unit side   0
	#	path point  1
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

		frameFronts = frames[:,:,0,:-1] # get a view for the front only
		
		# compute vector between all the points Di = roll( pathPoint, -1 ) - pathPoint
		unitDir = (np.roll( seedPathPointsView, -1, axis=1 )[:,:-1]) 
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
		firstFrameViewFront = frames[:,0,0,:-1]
		firstFrameViewUp = frames[:,0,1,:-1]
		firstFrameViewSide = frames[:,0,2,:-1]

		firstFrameViewUp[:] = [0,1,0] # we initialize Up with vector Y
		verticalStartBranch = np.where( np.abs( (firstFrameViewUp*firstFrameViewFront).sum(-1)) > 0.99999 )
		firstFrameViewUp[verticalStartBranch] = [1,0,0] # where it's vertical let's start with this vector instead

		# with a valid Up we can deduce the side then the up again to have a proper frame
		firstFrameViewSide[:] = np.cross( firstFrameViewFront, firstFrameViewUp )
		firstFrameViewSide /= np.sqrt( ((firstFrameViewSide**2).sum(-1)).reshape(-1,1) )

		firstFrameViewUp[:] = np.cross( firstFrameViewSide, firstFrameViewFront )

		# now apply the algorithm (Double Reflection) for each branch
		for i in range(1,self.maxPoints):
			frontPrevPoint = frames[:,i-1,0,:-1]
			upPrevPoint = frames[:,i-1,1,:-1]

			frontCurrentPoint = frames[:,i,0,:-1]
			upCurrentPoint = frames[:,i,1,:-1]
			sideCurrentPoint = frames[:,i,2,:-1]

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
		frames[:,:,3,:-1] = seedPathPointsView

	# now we can generate childs if we need
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
		self.timer.start()

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

		#sys.stderr.write('startSeedBranchesAttributes '+str(self.startSeedBranchesAttributes)+'\n')
		startSeedAttributes = np.array( [self.startSeedBranchesAttributes] ) 
		#sys.stderr.write('startSeedAttributes '+str(startSeedAttributes)+'\n')
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
		sys.stderr.write('circles '+str(circles.reshape(self.maxPoints, tubeSide,4))+'\n')
		sys.stderr.write('resultFrames[0] '+str(resultFrames[0])+'\n')
		sys.stderr.write('dot '+str( np.dot(circles.reshape(self.maxPoints, tubeSide,1,4),resultFrames[0]))+'\n')

		# avec v le vecteur et u la matrice
		# np.dot(v,u) donne les coordonnees transformee par u

		resultPoints = [ coord for array in resultFrames for k in range(len(array)) for point in (np.dot(circles[k],array[k])).tolist() for coord in point  ]

		sys.stderr.write('resultPoints '+str(resultPoints)+'\n')

		resultLoopingPoints = [] # TODO
		numLoopingLightningBranch = 0 # TODO

		average = self.timer.stop()

		sys.stderr.write('time '+str(average)+'\n')
		

		return self.outSoftwareFunc( resultLoopingPoints, numLoopingLightningBranch, resultPoints, numLightningBranch, self.maxPoints, tubeSide )

