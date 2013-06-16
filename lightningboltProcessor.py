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
		self.APVInputsMultiplier1 = np.zeros((1,lightningBoltProcessor.kAPV_MaxAttributes),np.float32)

		# has no variation along the path
		# O numchildens (nombre d'enfant genere obligatoirement)
		# 1 randNumChildrens (nombre d'enfant potentiellement genere aleatoirement en plus)

		self.VInputsGeneral = np.zeros((lightningBoltProcessor.kV_MaxAttributes),np.float32)
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
		self.APVInputsMultiplier1[0,num] = value

	def setVGeneral(self, num, value):
		self.VInputsGeneral[num] = value

	def generate( self, branchInputs, doGenerateChilds ):
		seedPath, APV, APVMult = branchInputs
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

	def process(self, pointList ):

		outBranchList = []
		maxGeneration = 3
		currentGeneration = 0

		# --- create the first datas (seedPath, attributes ) to process 
		startSeedPath = np.array( pointList, np.float32 )
		startAPV = self.APVInputs1
		startAPVMult = self.APVInputsMultiplier1

		toDo = [ (startSeedPath, startAPV, startAPVMult )]
		while len(toDo)>0:
			outBranch, childList = self.generate(toDo.pop(0),currentGeneration<maxGeneration)
			outBranchList.append(outBranch)
			toDo.extend(childList)
			return self.outSoftwareFunc(outBranchList)

		return self.outSoftwareFunc(outBranchList)

