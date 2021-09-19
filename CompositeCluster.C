/////////////////////////////////////////////////////
//Author:  Kellie McGuire     kellie@kelliejensen.com
//
//Takes as input .root files generated from the cluster
//finder method of the pysimdamicm software and generates a
//composite cluster within the specified energy range
//////////////////////////////////////////////////////


#include "TMap.h"
#include "TChain.h"
#include "TStyle.h"
#include "TObject.h"
#include "TROOT.h"
#include "TString.h"
#include <iostream>
#include <vector>
using namespace std;


void CompositeCluster(){

    TChain* j = new TChain("clustersRec");
    j->Add("*.root");

    //TFile *f = TFile::Open("trial_1/panaSKImg_clustersRec_Fe_55_02.root","READ");
    //TTree *t; j->GetObject("clustersRec",t);

    ////These lines create the dictionaries for the vector<vector<type> > classes
    gInterpreter->GenerateDictionary("vector<vector<int> >", "vector");
    gInterpreter->GenerateDictionary("vector<vector<float> >", "vector");


    std::vector<vector<int> > *pixels_x = 0;
    std::vector<vector<int> > *pixels_y = 0;
    std::vector<vector<float> > *pixels_E = 0;
    std::vector<float> *QmaxX = 0;
    std::vector<float> *QmaxY = 0;
    std::vector<float> *Energy = 0;

    j->SetBranchAddress("QmaxX",&QmaxX);
    j->SetBranchAddress("QmaxY",&QmaxY);
    j->SetBranchAddress("pixels_x",&pixels_x);
    j->SetBranchAddress("pixels_y",&pixels_y);
    j->SetBranchAddress("pixels_E",&pixels_E);
    j->SetBranchAddress("Energy",&Energy);


    ////Create a new canvas.
    TCanvas *c1 = new TCanvas("c1","",200,10,700,500);

    ////Create histograms
    TH2F *h1 = new TH2F("h1","",20,-10,10,20,-10,10);
    TH2F *h2 = new TH2F("h2","",3000,0,3000,2000,0,2000);


    ////loop over the files in the chain
    for (Int_t i = 0; i < j->GetEntries(); i++){

      //Long64_t tentry = j->LoadTree(i);
      j->GetEntry(i);

      ////loop over clusters in clustersRec
      for (UInt_t j = 0; j < pixels_x->size(); ++j) {

        ////loop over pixels in cluster j
        for (UInt_t k = 0; k < pixels_x->at(j).size(); ++k){
          ////select clusters w/in given energy range
          if(Energy->at(j)<6.5 && Energy->at(j)>4.5){
            h1->Fill(pixels_x->at(j).at(k)-QmaxX->at(j),pixels_y->at(j).at(k)-QmaxY->at(j),pixels_E->at(j).at(k));
            }

          h2->Fill(pixels_x->at(j).at(k),pixels_y->at(j).at(k),pixels_E->at(j).at(k));
           // cout <<"pixels_x["<<j<<"]["<< k << "]: "<< pixels_x->at(j).at(k)<<endl;
           // cout <<"pixels_y["<<j<<"]["<< k << "]: "<< pixels_y->at(j).at(k)<<endl;
           // cout <<"pixels_E["<<j<<"]["<< k << "]: "<< pixels_E->at(j).at(k)<<endl;

        }


      }
      ////Draw composite cluster
      h1->Draw("Colz");
      ////Draw image with all clusters
      //h2->Draw("Colz");
      c1->Modified();
      c1->Update();

    }

    j->ResetBranchAddresses();
}
