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
- generate full rotation matrix instead of just vector on sphere ( so we can transmit the frame and we don't need to calculate the start Up for every branch or maybe not)
- faire une class de noise encore plus general, pouvant generer plusieure suite de nombre pour plusieures seeds
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

def enum(*sequential, **named):
	enums = dict(zip(sequential, range(len(sequential))), **named)
	return type('Enum', (), enums)

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

def dot2d(g, x, y):
	return g[0]*x + g[1]*y

_CONST1 = 0.5 * (math.sqrt(3.0) - 1.0)
_CONST2 = (3.0 - math.sqrt(3.0)) / 6.0

permutation = np.array([151,160,137,91,90,15, 
	131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23, 
	190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33, 
	88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,134,139,48,27,166, 
	77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244, 
	102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,169,200,196, 
	135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,124,123, 
	5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42, 
	223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,172,9, 
	129,22,39,253,9,98,108,110,79,113,224,232,178,185,112,104,218,246,97,228, 
	251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107, 
	49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254, 
	138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180] )

class simpNoise():
	"""The gradients are the midpoints of the vertices of a cube."""
	_grad3 = np.array([
	[1,1,0], [-1,1,0], [1,-1,0], [-1,-1,0],
	[1,0,1], [-1,0,1], [1,0,-1], [-1,0,-1],
	[0,1,1], [0,-1,1], [0,1,-1], [0,-1,-1]], np.int)

	def __init__(self, seed=0):
		#sys.stderr.write('simpNoise.permutation'+str(permutation) +'\n' )

		self.nprndState = np.random.RandomState( int(seed) )

		np.random.seed( int(seed) )
		#test = np.array([5,7,2,4], np.int )
		#np.random.shuffle( test )

		#sys.stderr.write('test '+str(test) +'\n' )

		#sys.stderr.write('self.nprndState.shuffle(permutation) '+str(self.nprndState.shuffle(permutation)) +'\n' )
		self.permX = permutation.copy()
		self.permY = permutation.copy()
		self.permZ = permutation.copy()
		self.permW = permutation.copy()
		self.nprndState.shuffle(self.permX)
		self.nprndState.shuffle(self.permY)
		self.nprndState.shuffle(self.permZ)
		self.nprndState.shuffle(self.permW)

		self.permX = np.tile( self.permX , 2)
		#sys.stderr.write('self.permX'+str(self.permX) +'\n' )

		self.permY = np.tile( self.permY , 2)
		self.permZ = np.tile( self.permZ , 2)
		self.permW = np.tile( self.permY , 2)

	
	def in2D_outVect( self, noiseCoords ):
		# Noise contributions from the three corners
		contrib = np.zeros( (len(noiseCoords),3,3) , np.float) # samples, coord(X,Y,Z), contrib
		
		# Skew the input space to determine which simplex cell we're in
		F2 = _CONST1#0.5 * (math.sqrt(3.0) - 1.0)
		# Hairy skew factor for 2D

		s2 = noiseCoords.sum(-1)*F2#------------
		ij2 = noiseCoords.copy()#------------
		ij2.transpose((1,0))[:] += s2#------------
		ij2 = ij2.astype(int)#------------

		G2 = _CONST2#(3.0 - math.sqrt(3.0)) / 6.0
		t2 = ij2.sum(-1).astype(float) * G2 #------------

		# Unskew the cell origin back to (x,y) space
		XY02 = ij2.copy().astype(float)#------------
		XY02.transpose((1,0))[:] -= t2#------------

		# The x,y distances from the cell origin
		coord02 = noiseCoords - XY02#------------

		# For the 2D case, the simplex shape is an equilateral triangle.
		# Determine which simplex we are in.
		
		# Offsets for second (middle) corner of simplex in (i,j) coords
		positiveIds2 = np.where( coord02[:,0]>coord02[:,1] )#------------
		ij12 = np.tile( [0,1], (len(noiseCoords),1))#------------
		ij12[positiveIds2] = [1,0]#------------

		# A step of (1,0) in (i,j) means a step of (1-c,-c) in (x,y), and
		# a step of (0,1) in (i,j) means a step of (-c,1-c) in (x,y), where
		# c = (3-sqrt(3))/6
		coord12 = coord02 - ij12 + G2#------------
		#x1 = x0 - i1 + G2       # Offsets for middle corner in (x,y) unskewed coords
		#y1 = y0 - j1 + G2

		coord22 = coord02.copy() + (2.0 * G2 - 1.0)#------------
		#x2 = x0 + (2.0 * G2 - 1.0) # Offsets for last corner in (x,y) unskewed coords
		#y2 = y0 + (2.0 * G2 - 1.0)

		Gcoord02 = np.array( [coord02,coord02,coord02] )
		Gcoord02 = Gcoord02.transpose( (1,0,2) )
		#sys.stderr.write('Gcoord02 '+str(Gcoord02) +'\n' )

		Gcoord12 = np.array( [coord12,coord12,coord12] )
		Gcoord12 = Gcoord12.transpose( (1,0,2) )
		#sys.stderr.write('Gcoord12 '+str(Gcoord12) +'\n' )

		Gcoord22 = np.array( [coord22,coord22,coord22] )
		Gcoord22 = Gcoord22.transpose( (1,0,2) )
		#sys.stderr.write('Gcoord22 '+str(Gcoord22) +'\n' )

		# Work out the hashed gradient indices of the three simplex corners
		ii = ij2[:,0] & 255# i & 255
		jj = ij2[:,1] & 255#j & 255

		#sys.stderr.write('jj '+str(jj) +'\n' )
		#sys.stderr.write('jj+j1 '+str(jj+j1) +'\n' )

		#permMatrix = np.array( [self.permX, self.permY,self.permZ ] )

	 	#sys.stderr.write('ii'+str(ii) +'\n' )
	 	coordMatrix = np.tile( ii, (3,3,1) )
	 	coordMatrix[:,1] += ij12[:,0]
	 	coordMatrix[:,2] += 1
		
		coordMatrix[0] += self.permX[jj]
		coordMatrix[1] += self.permY[jj+ij12[:,1]]
		coordMatrix[2] += self.permZ[jj+1]

		#coordMatrix.transpose( (1,0))  += permMatrix[ jj, jj+ ij12[:,1], jj+1 ]
	 	#sys.stderr.write('coordMatrix'+str(coordMatrix) +'\n' )

		#testCoordX = np.array( [ ii+self.permX[jj], ii+ij12[:,0]+self.permX[jj+ij12[:,1]] , ii+1+self.permX[jj+1] ] )
		#testCoordY = np.array( [ ii+self.permY[jj], ii+ij12[:,0]+self.permY[jj+ij12[:,1]] , ii+1+self.permY[jj+1] ] )
		#testCoordZ = np.array( [ ii+self.permZ[jj], ii+ij12[:,0]+self.permZ[jj+ij12[:,1]] , ii+1+self.permZ[jj+1] ] )
	 	#sys.stderr.write('testCoordX'+str(testCoordX) +'\n' )
	 	#sys.stderr.write('testCoordY'+str(testCoordY) +'\n' )
	 	#sys.stderr.write('testCoordZ'+str(testCoordZ) +'\n' )

		gi0 = np.array( [ self.permX[ii+self.permX[jj]], self.permY[ii+self.permY[jj]], self.permZ[ii+self.permZ[jj]] ] ) % 12
		#sys.stderr.write('gi0'+str(gi0) +'\n' )
		gi0 = gi0.T
		#sys.stderr.write('gi0.T'+str(gi0) +'\n' )

		gi1 = np.array( [ self.permX[ii+ij12[:,0]+self.permX[jj+ij12[:,1]]], self.permY[ii+ij12[:,0]+self.permY[jj+ij12[:,1]]], self.permZ[ii+ij12[:,0]+self.permZ[jj+ij12[:,1]]] ] ) % 12
		#sys.stderr.write('gi1'+str(gi1) +'\n' )
		gi1 = gi1.T
		#sys.stderr.write('gi1T'+str(gi1) +'\n' )

		gi2 = np.array( [ self.permX[ii+1+self.permX[jj+1]], self.permY[ii+1+self.permY[jj+1]], self.permZ[ii+1+self.permZ[jj+1]]] ) % 12
		gi2 = gi2.T

		# Calculate the contribution from the three corners
		#t0 = 0.5 - x0**2 - y0**2
		t0 = .5 - (coord02**2).sum(-1)
		indPos = np.where( t0>=0.0 )
		t0Pos = t0[indPos]**3
		Gt0Pos = np.array( [t0Pos, t0Pos, t0Pos] )
		Gt0Pos = Gt0Pos.T		
		t = (simpNoise._grad3[ gi0[indPos] ])[:,:,:-1]
		contrib[indPos,:,0] = Gt0Pos* (t*Gcoord02[indPos]).sum(-1)

		t1 = .5 - (coord12**2).sum(-1)
		indPos = np.where( t1>=0.0 )
		t1Pos = t1[indPos]**3
		Gt1Pos = np.array( [t1Pos, t1Pos, t1Pos] )
		Gt1Pos = Gt1Pos.T
		t = (simpNoise._grad3[ gi1[indPos] ])[:,:,:-1]
		contrib[indPos,:,1] = Gt1Pos* (t*Gcoord12[indPos]).sum(-1)
		#sys.stderr.write('contrib2 '+str( contrib[indPos,:,1]  ) +'\n' )

		t2 = .5 - (coord22**2).sum(-1)
		indPos = np.where( t2>=0.0 )
		t2Pos = t2[indPos]**3
		Gt2Pos = np.array( [t2Pos, t2Pos, t2Pos] )
		Gt2Pos = Gt2Pos.T
		t = (simpNoise._grad3[ gi2[indPos] ])[:,:,:-1]
		contrib[indPos,:,2] = Gt2Pos* (t*Gcoord22[indPos]).sum(-1)
		#sys.stderr.write('contrib3 '+str( contrib[indPos,:,2]  ) +'\n' )

		return 70.0*(contrib.sum(-1))


	def raw_noise_2dVector(self, x, y):
		"""2D Raw Simplex noise."""
		# Noise contributions from the three corners
		n0 = [0]*3
		n1 = [0]*3
		n2 = [0]*3

		#n0, n1, n2 = 0.0, 0.0, 0.0

		# Skew the input space to determine which simplex cell we're in
		F2 = _CONST1#0.5 * (math.sqrt(3.0) - 1.0)
		# Hairy skew factor for 2D
		s = (x + y) * F2
		i = int(x + s)
		j = int(y + s)

		G2 = _CONST2#(3.0 - math.sqrt(3.0)) / 6.0
		t = float(i + j) * G2
		# Unskew the cell origin back to (x,y) space
		X0 = i - t
		Y0 = j - t
		# The x,y distances from the cell origin
		x0 = x - X0
		y0 = y - Y0

		# For the 2D case, the simplex shape is an equilateral triangle.
		# Determine which simplex we are in.
		i1, j1 = 0, 0 # Offsets for second (middle) corner of simplex in (i,j) coords
		if x0 > y0: # lower triangle, XY order: (0,0)->(1,0)->(1,1)
			i1 = 1
			j1 = 0
		else:        # upper triangle, YX order: (0,0)->(0,1)->(1,1)
			i1 = 0
			j1 = 1

		# A step of (1,0) in (i,j) means a step of (1-c,-c) in (x,y), and
		# a step of (0,1) in (i,j) means a step of (-c,1-c) in (x,y), where
		# c = (3-sqrt(3))/6
		x1 = x0 - i1 + G2       # Offsets for middle corner in (x,y) unskewed coords
		y1 = y0 - j1 + G2
		x2 = x0 - 1.0 + 2.0 * G2  # Offsets for last corner in (x,y) unskewed coords
		y2 = y0 - 1.0 + 2.0 * G2

		# Work out the hashed gradient indices of the three simplex corners
		ii = int(i) & 255
		jj = int(j) & 255
		gi0 = [0]*3
		gi1 = [0]*3
		gi2 = [0]*3
		gi0[0] = self.permX[ii+self.permX[jj]] % 12
		gi1[0] = self.permX[ii+i1+self.permX[jj+j1]] % 12
		gi2[0] = self.permX[ii+1+self.permX[jj+1]] % 12

		gi0[1] = self.permY[ii+self.permY[jj]] % 12
		gi1[1] = self.permY[ii+i1+self.permY[jj+j1]] % 12
		gi2[1] = self.permY[ii+1+self.permY[jj+1]] % 12

		gi0[2] = self.permZ[ii+self.permZ[jj]] % 12
		gi1[2] = self.permZ[ii+i1+self.permZ[jj+j1]] % 12
		gi2[2] = self.permZ[ii+1+self.permZ[jj+1]] % 12

		# Calculate the contribution from the three corners
		t0 = 0.5 - x0*x0 - y0*y0
		if t0 < 0:
			pass
		else:
			t0 *= t0*t0
			n0[0] = t0 * dot2d(simpNoise._grad3[gi0[0]], x0, y0)
			n0[1] = t0 * dot2d(simpNoise._grad3[gi0[1]], x0, y0)
			n0[2] = t0 * dot2d(simpNoise._grad3[gi0[2]], x0, y0)
			#sys.stderr.write('contrib1 '+str( n0[0]  ) +' '+str(n0[1])+' '+str(n0[2]) +'\n' )

		t1 = 0.5 - x1*x1 - y1*y1
		if t1 < 0:
			pass
		else:
			t1 *= t1*t1
			n1[0] = t1 * dot2d(simpNoise._grad3[gi1[0]], x1, y1)
			n1[1] = t1 * dot2d(simpNoise._grad3[gi1[1]], x1, y1)
			n1[2] = t1 * dot2d(simpNoise._grad3[gi1[2]], x1, y1)
			#sys.stderr.write('contrib2 '+str( n1[0]  ) +' '+str(n1[1])+' '+str(n1[2]) +'\n' )

		t2 = 0.5 - x2*x2-y2*y2
		if t2 < 0:
			pass
		else:
			t2 *= t2*t2
			n2[0] = t2 * dot2d(simpNoise._grad3[gi2[0]], x2, y2)
			n2[1] = t2 * dot2d(simpNoise._grad3[gi2[1]], x2, y2)
			n2[2] = t2 * dot2d(simpNoise._grad3[gi2[2]], x2, y2)
			#sys.stderr.write('contrib3 '+str( n2[0]  ) +' '+str(n2[1])+' '+str(n2[2]) +'\n' )

		# Add contributions from each corner to get the final noise value.
		# The result is scaled to return values in the interval [-1,1].
		return [70.0 * (n0[0] + n1[0] + n2[0]), 70.0 * (n0[1] + n1[1] + n2[1]), 70.0 * (n0[2] + n1[2] + n2[2]) ]


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
	theta = np.arccos( np.clip( aAmplitude*aAmplitudeRand - 2.0*aElevation + 1.0, -1.0,1.0 ))
	sphPoints[:,1] = sphPoints[:,2] = np.sin( theta )
	sphPoints[:,0] =  np.cos( theta )
	sphPoints[:,1] *= np.cos( aPhi )
	sphPoints[:,2] *= np.sin( aPhi )
	return sphPoints


class lightningBoltProcessor:
	# different types of values
		# default : On value for the overall path
		# AP : Along Path, it's an array descripbing the variation along the full path of a branch

	# different scopes
		# default : the values apply to all the branches
		# BR : Per Branch, means each branch may have different values for this attributes
		# GEN : Per Generation, means all branch of the same generation share this value

	# different transmission behavior
		# default : there is transfer value that is multiply to it before transmitting to next generation
		# SPE : the transmission to next generation is Special

	# Along Path values Per Branch (attributes transmitted to children by mult)
	eAPBR = enum('radius', 'intensity', 'childLength', 'max')

	# Special Values Per Branch enum (no transmitted by function)
	# seedBranching is the seed that drive the number of random childs and their position along the path
	# seedShape is the seed that drive the shape of the tube chaos
	eSPEBR = enum('seedBranching', 'seedShape', 'max')

	# Values Depending on Generation and that are transmitted to the next generation by multiply vector
	eGEN = enum('numChildrens', 'randNumChildrens', 'branchingTimeMult', 'shapeTimeMult', 'shapeFrequency', 'offset', 'max')

	# Along Path values Special ( No Transfert to Child )
	eAPSPE = enum('elevation', 'elevationRand', 'childProbability', 'offset', 'max')


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
		self.APBRInputs1 = None
		# this is a vector with all the multiplicator to apply to APVInputs
		self.APBRInputsMultiplier1 = np.zeros((lightningBoltProcessor.eAPBR.max),np.float32)
		self.APBRTransfert = np.ones(lightningBoltProcessor.eAPBR.max,np.float32) # mults for transfert of attribute from parent to child

		# Per Branch Special values (is not transmitted by function to child)
		# 0 seedBranching (the seed that decided the number of childrens and the parameters of each one)
		self.SPEBRInputs = np.zeros((lightningBoltProcessor.eSPEBR.max),np.float32)

		# has no variation along the path is transmitted to child
		# O numchildens (nombre d'enfant genere obligatoirement)
		# 1 randNumChildrens (nombre d'enfant potentiellement genere aleatoirement en plus)
		# 2 branchingTimeMult (Multiplicator to timeBranching used to generate the childs number and params)
		self.GENInputs = np.zeros((lightningBoltProcessor.eGEN.max),np.float32)
		self.GENTransfert = np.ones(lightningBoltProcessor.eGEN.max,np.float32) # mult for transfert of attribute from parent to child

		self.APSPEInputs1 = None


		#for testing
		#self.setValue(lightningBoltProcessor.eV.numChildrens, 2.0 ) 
		#self.setValue(lightningBoltProcessor.eV.randNumChildrens, 2.0 ) 
		#self.setGenericValue(lightningBoltProcessor.eGV.seedBranching, 437 )
		#self.setGenericValue(lightningBoltProcessor.eGV.seedShape, 20 )
		#self.setAPVTransfert( lightningBoltProcessor.eAPV.radius, 0.25)

	def setDetail( self, detail):
		if detail == self.detail:
			return
		self.detail = detail
		self.maxPoints = 2 + 2**detail
		self.APBRInputs1 = np.zeros((self.maxPoints,lightningBoltProcessor.eAPBR.max),np.float32)
		self.APSPEInputs1 = np.zeros((self.maxPoints,lightningBoltProcessor.eAPSPE.max),np.float32)

# Along Path Values Per Branch
	def getAPVariation1(self, num):
		''' to get the array specific to one attribute '''
		return self.APBRInputs1[:,num]

	def setAPVMult1(self, num, value):
		self.APBRInputsMultiplier1[num] = value
	def setAPVTransfert(self, num, value):
		self.APBRTransfert[num] = value

# Generation Dependent
	def setValue(self, num, value):
		self.GENInputs[num] = value
	def setValueTransfert(self, num, value):
		self.GENTransfert[num] = value

# Along Path Special Values
	def getAPSPE1(self, num):
		''' to get the array specific to one attribute '''
		return self.APSPEInputs1[:,num]
		
# Specials Values Per Branch
	def setSpecialBranchValue(self, num, value):
		self.SPEBRInputs[num] = value
#----------	

	def initializeProcessor( self ):
		self.startSeedPoints = []
		self.numStartSeedBranch = 0
		self.startSeedBranchesAPVMult = []
		self.startSeedBranchesSPEV = []

	def addSeedPointList( self, pointList):
		self.startSeedPoints.extend( pointList )
		#self.startSeedPoints.append([0,0,0]) # add an extra blank point
		self.numStartSeedBranch =  self.numStartSeedBranch + 1
		self.startSeedBranchesAPVMult.append(self.APBRInputsMultiplier1.reshape(1,-1))
		self.startSeedBranchesSPEV.append(self.SPEBRInputs)

#sys.stderr.write('seedPathPointsView '+str(seedPathPointsView)+'\n')
	def generate( self, batch, APBranchV, APSpeV, branchingTime, shapeTime, isLooping, doGenerateChilds ):
		# unpack
		batchSize, seedPath, APBranchMults, GenValues, SPEBRValues = batch

		sys.stderr.write('APBranchV '+str(APBranchV)+'\n')
		#sys.stderr.write('batchSize '+str(batchSize)+'\n')
		#sys.stderr.write('seedPath '+str(seedPath)+'\n')
		testBranchMult = APBranchMults.reshape(batchSize,1,-1)
		sys.stderr.write('APBranchMults '+str(testBranchMult)+'\n')
		#sys.stderr.write('Values '+str(Values)+'\n')
		#totalBatchPoints = len(seedPath) # this should be self.maxPoints*batchSize

		# duplicate the Path variation values to have one for each branch
		APVs = np.tile(APBranchV,(batchSize,1,1))

		# get the final values at each path point ( the value from the ramp multiply by the multiplicator )
		APArray = APVs * APBranchMults # the attribute path Array contain every attribute value at every point
		sys.stderr.write('Target '+str(APArray)+'\n')
		#APArray = (testBranchMult*APBranchV)[np.newaxis,:]
		sys.stderr.write('test '+str(testBranchMult*APBranchV)+'\n')
		#sys.stderr.write('APArray '+str(APArray)+'\n')

		# get the a random vector for every path point (that will offset the position of seedPath)
		#simp = simpNoise(0)


		#sys.stderr.write('shapeTime '+str(shapeTime)+'\n')
		#sys.stderr.write('Values[:,lightningBoltProcessor.eV.shapeTimeMult] '+str(Values[:,lightningBoltProcessor.eV.shapeTimeMult])+'\n')
		finalShapeTime = GenValues[lightningBoltProcessor.eGEN.shapeTimeMult]*shapeTime
		#sys.stderr.write('finalShapeTime '+str(finalShapeTime)+'\n')
		

		#res = simp.in2D_outVect(x, y, noisePos)
		#sys.stderr.write('res '+str(res)+'\n')
		#simp = simpNoise(0)
		#for i in range(len(y)) :
			#res2 = simp.raw_noise_2dVector( shapeTime, y[i] )
			#sys.stderr.write('res2 '+str(res2)+'\n')

		# For generating childs
		SPEBRValues[:,lightningBoltProcessor.eSPEBR.seedBranching] += np.floor( GenValues[lightningBoltProcessor.eGEN.branchingTimeMult]*branchingTime )
		allChildsBranchParentId = []
		allRandValues = []

		offsetArray = np.empty( (batchSize,self.maxPoints,3), np.float32 )
		freq = GenValues[lightningBoltProcessor.eGEN.shapeFrequency]
		noisePos = np.empty( (self.maxPoints,2), np.float32 )
		noisePos[:,0] = finalShapeTime

		for i in range(batchSize):			
			simp = simpNoise( SPEBRValues[i,lightningBoltProcessor.eSPEBR.seedShape] )			
			branchLength = APBranchMults[0,i,0,lightningBoltProcessor.eAPBR.childLength]
			#sys.stderr.write('branchLength '+str(branchLength)+'\n')
			noisePos[:,1] = np.linspace(0.0,branchLength*freq,self.maxPoints)

			offsetArray[i] = simp.in2D_outVect(noisePos)

			# childs calculations
			if not doGenerateChilds:
				continue

			randBranch = nRand(SPEBRValues[i,lightningBoltProcessor.eSPEBR.seedBranching])
			branchChildNumber = int( GenValues[lightningBoltProcessor.eGEN.numChildrens] + randBranch.randInt(0, GenValues[lightningBoltProcessor.eGEN.randNumChildrens] ) )
			epsilon = 0.00001
			randArray = randBranch.npaRand(epsilon, 2.0-epsilon, (branchChildNumber, 5) )
			allRandValues.extend( randArray.tolist() )
			allChildsBranchParentId.extend( [i]*branchChildNumber )


			#sys.stderr.write('offsetArray[i] '+str(offsetArray[i])+'\n')


		#offsetArray = (arndSphereVect(totalBatchPoints)).reshape(batchSize,self.maxPoints,3)
		#sys.stderr.write('offsetArray '+str(offsetArray)+'\n')
		

		# multiply all the random vector by the offset multiplicator at each point
		#test = APArray[:,:,:,lightningBoltProcessor.kAPV_Offset].reshape(1,1,batchSize,self.maxPoints,-1)
		#sys.stderr.write('test '+str(test)+'\n')

		seedPathPointsView = seedPath.reshape(batchSize,self.maxPoints,3)
		seedPathPointsView += GenValues[lightningBoltProcessor.eGEN.offset]*offsetArray*APSpeV[:,lightningBoltProcessor.eAPSPE.offset].reshape(1,-1,1)
		#seedPathPointsView += (offsetArray*APArray[:,:,:,lightningBoltProcessor.eAPBR.offset].reshape(1,1,batchSize,self.maxPoints,-1))[0,0]
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
		frames[:,:,:-1,:-1] *= APArray[0,:,:,lightningBoltProcessor.eAPBR.radius,np.newaxis,np.newaxis]

		# load branch points in frames to complete the transformation matrix (we are done with the frames!)
		frames[:,:,:-1,3] = seedPathPointsView

	# now we can generate childs if we need
		childsBatch = None
		totalChildNumber = len(allChildsBranchParentId)			
		if totalChildNumber>0 :
			aChildRandomsArray = np.array( allRandValues, np.float32 )
			aChildRandomsArray = aChildRandomsArray.T

			aChildRandomsArray[0,:]*=.5*float(self.maxPoints-1) # bring the parameters back to [0,maxPoints]
			indexT0 = aChildRandomsArray[0,:].astype(np.uint32)

			blend = aChildRandomsArray[0,:] # VIEW
			blend -= indexT0
			blend = blend.reshape(-1,1)
		# Get the Along Path Per Generation Values for the elevation and elevationRand for all theses childs
			APGenT0 = APSpeV[indexT0,lightningBoltProcessor.eAPSPE.elevation:lightningBoltProcessor.eAPSPE.elevationRand+1]
			APGenChildElevationAndElevationRand =  APGenT0 + blend*( APSpeV[indexT0+1,lightningBoltProcessor.eAPSPE.elevation:lightningBoltProcessor.eAPSPE.elevationRand+1] - APGenT0 )

		# Now get the Along Path Per Branch Values
			aChildParentId = np.array(allChildsBranchParentId, np.float32)
			indexT0 += aChildParentId*self.maxPoints # indices must now point to the correct parent branch

			#sys.stderr.write('aChildRandomsArray[0] '+str(aChildsParamGlobal)+'\n')
			#sys.stderr.write('allChildsBranchParentId '+str(allChildsBranchParentId)+'\n')

			APVView = APArray.reshape(-1,lightningBoltProcessor.eAPBR.max)
			#sys.stderr.write('APArray '+str(APArray)+'\n')
			#sys.stderr.write('allChildsParams '+str(allChildsParams)+'\n')
			#blend = aChildsParamGlobal #aChildRandomsArray[0,:]  #np.array(allChildsParams, np.float32)
			#sys.stderr.write('blend '+str(blend)+'\n')
			##indexT0 = aChildsParamGlobal.astype(np.uint32)# aChildRandomsArray[0,:].astype(np.uint32) #np.array(allChildsParams, np.uint32)
			APVViewT0 = APVView[indexT0]
			#sys.stderr.write('indexT0 '+str(indexT0)+'\n')
			#sys.stderr.write('indexT0+1 '+str(indexT0+1)+'\n')
			##blend -= indexT0
			##blend = blend.reshape(-1,1)
			#sys.stderr.write('blend '+str(blend)+'\n')
			#sys.stderr.write('APVView[indexT0]'+str(APVView[indexT0])+'\n')
			#sys.stderr.write('blend*APVView[indexT0]'+str(blend*APVView[indexT0])+'\n')
			APVMultChilds = APVViewT0 + blend*( APVView[indexT0+1] - APVViewT0 )
			APVMultChilds *= self.APBRTransfert # transfert Along Path Branch multiply

			#ValuesChilds = Values[allChildsBranchParentId]
			ValuesNextGen = GenValues*self.GENTransfert # transfert Values multiply

			SPEBRValuesChilds = np.empty( (totalChildNumber,lightningBoltProcessor.eSPEBR.max ), np.float32  )
			#SPEBRValuesChilds = SPEBRValues[allChildsBranchParentId]

			# set the seeds of the childs
			aAllSeeds = aChildRandomsArray[3:5,:]
			aAllSeeds *= 500000.0
			aAllSeeds %= 1000000

			#sys.stderr.write('GValuesChilds '+str(GValuesChilds)+'\n')
			SPEBRValuesChilds[:,lightningBoltProcessor.eSPEBR.seedBranching] = aAllSeeds[0]
			SPEBRValuesChilds[:,lightningBoltProcessor.eSPEBR.seedShape] = aAllSeeds[1]
			#sys.stderr.write('GValuesChilds[seedBranching] '+str(GValuesChilds[:,lightningBoltProcessor.eGV.seedBranching])+'\n')
			#sys.stderr.write('GValuesChilds[seedShape] '+str(GValuesChilds[:,lightningBoltProcessor.eGV.seedShape])+'\n')
			

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
			#aPhiAndRandAmplitude = 	aChildRandomsArray[:2,:]
			#aPhiAndRandAmplitude = np.array( allRandValuesForSpherePt)
			aPhi = aChildRandomsArray[1,:] #aPhiAndRandAmplitude[0]
			aAmplitudeRand = aChildRandomsArray[2,:] #aPhiAndRandAmplitude[1]
			aPhi*=np.pi 
			aAmplitudeRand-=1.0
			 
			randomVectors = npaRandSphereElevation( APGenChildElevationAndElevationRand[:,0], APGenChildElevationAndElevationRand[:,1], aPhi, aAmplitudeRand, aAmplitudeRand.size  )
			#randomVectors = npaRandSphereElevation( APVMultChilds[:,lightningBoltProcessor.eAPBR.elevation], APVMultChilds[:,lightningBoltProcessor.eAPBR.elevationRand],aPhi, aAmplitudeRand, aAmplitudeRand.size  )


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
			#sys.stderr.write('childFrames'+str(childFrames) +'\n' )
			#sys.stderr.write('APVMultChilds[:,lightningBoltProcessor.eAPV.childLength]'+str(APVMultChilds[:,lightningBoltProcessor.eAPV.childLength]) +'\n' )
			childFrames *= APVMultChilds[:,lightningBoltProcessor.eAPBR.childLength,np.newaxis]
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

			childsBatch = ( totalChildNumber, seedPathChilds.reshape(-1,3) , APVMultChilds.reshape(1,totalChildNumber,1,-1), ValuesNextGen, SPEBRValuesChilds )
			#batchSize, seedPath, APVMults, Values, GValues = batch
			
			#sys.stderr.write('APArray[0,:,:,lightningBoltProcessor.eAPV.intensity]'+str(APArray[0,:,:,lightningBoltProcessor.eAPV.intensity]) +'\n' )


		return frames, APArray[0,:,:,lightningBoltProcessor.eAPBR.intensity], childsBatch

	def process(self, maxGeneration, branchingTime, shapeTime, tubeSide ):
		self.timer.start()

		currentGeneration = 0

		firstGenerationAPBranch  = self.APBRInputs1
		normalAPBranch = self.APBRInputs1
		firstGenerationAPSpe  = self.APSPEInputs1
		normalAPSpe = self.APSPEInputs1

		# circle template for tube extrusion
		#tubeSide = 4
		pointsOfCircle = np.zeros( (4,tubeSide), np.float32 )
		pointsOfCircle[1] = np.sin(np.linspace(1.5*np.pi,-np.pi*.5, tubeSide, endpoint=False))
		pointsOfCircle[2] = np.cos(np.linspace(1.5*np.pi,-np.pi*.5, tubeSide, endpoint=False))
		pointsOfCircle[3] = np.ones(tubeSide)
		pointsOfCircle = pointsOfCircle.T

		# intensity color template
		colorTemplate = np.ones( (tubeSide,4), np.float32 )

		# --- create the first datas (seedPath, attributes ) to process 
		startSeedPath = np.array( self.startSeedPoints )

		#sys.stderr.write('startSeedBranchesAttributes '+str(self.startSeedBranchesAttributes)+'\n')
		startSeedAPVMult = np.array( [self.startSeedBranchesAPVMult] ) 
		#startSeedValues = np.array( self.startSeedBranchesV )
		startSeedBranchSpeValues = np.array( self.startSeedBranchesSPEV )
		

		APBranch = firstGenerationAPBranch
		APSpe = firstGenerationAPSpe

		# The result returned will be points in the form of a huge list of coordinate, there will be 2 lists to separate ring and non ring lightning
		resultLoopingFrames = []
		numLoopingLightningBranch = 0
		resultFrames = []
		numLightningBranch = 0
		isLooping = False

		resultIntensities = []

		batchSize = self.numStartSeedBranch
		batch = batchSize, startSeedPath, startSeedAPVMult, self.GENInputs, startSeedBranchSpeValues
		while batch is not None:
			outFrames, outIntensities, childBatch = self.generate( batch, APBranch, APSpe, branchingTime, shapeTime, isLooping, currentGeneration<maxGeneration)

			if isLooping :
				resultLoopingFrames.append(outFrames.reshape(-1,4,4))
				numLoopingLightningBranch += batch[0]
			else:
				resultFrames.append(outFrames.reshape(-1,4,4))
				numLightningBranch += batch[0]
				resultIntensities.append(outIntensities.reshape(-1))

			batch = childBatch
			# set the generation to the general values
			APBranch = normalAPBranch
			APSpe = normalAPSpe
			isLooping = False # only the first generation can loop
			currentGeneration = currentGeneration + 1

		# here deduce points from frame and circles and add them to resultPoints
		# transform a circle of point for each frame
		resultPoints = [ coord for array in resultFrames for coord in ((np.inner(array,pointsOfCircle)).transpose(0,2,1)).reshape(-1).tolist() ]

		#sys.stderr.write('colorTemplate '+str(colorTemplate)+'\n')
		#sys.stderr.write('resultIntensities[0] '+str(resultIntensities[0][:,np.newaxis,np.newaxis])+'\n')
		#sys.stderr.write('resultIntensities[0]*colorTemplate '+str(colorTemplate*resultIntensities[0][:,np.newaxis,np.newaxis])+'\n')
		resultIntensityColors = [ coord for array in resultIntensities for coord in (colorTemplate*resultIntensities[0][:,np.newaxis,np.newaxis]).reshape(-1).tolist() ]
		#resultPoints = [ coord for array in resultFrames for k in range(len(array)) for point in (np.dot(circles[k],array[k])).tolist() for coord in point  ]

		#sys.stderr.write('resultPoints '+str(resultPoints)+'\n')
		#sys.stderr.write('resultIntensityColors '+str(resultIntensityColors)+'\n')
		#sys.stderr.write('num points '+str(len(resultPoints))+'\n')

		resultLoopingPoints = [] # TODO
		numLoopingLightningBranch = 0 # TODO

		average = self.timer.stop()

		sys.stderr.write('time '+str(average)+'\n')

		return self.outSoftwareFunc( resultLoopingPoints, numLoopingLightningBranch, resultPoints, numLightningBranch, resultIntensityColors, self.maxPoints, tubeSide )

