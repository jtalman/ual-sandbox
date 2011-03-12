//  UAL::OpticsCalculator& optics = UAL::OpticsCalculator::getInstance();
//  Teapot* teapot = optics.m_teapot;
//  PacSurveyData surveyData;
//  TeapotElement te = teapot->element(m_i0);
//  std::string nameInput;
//  nameInput=te.getDesignName();
//  std::cout << "nameInput=te.getDesignName() " << nameInput << "\n";
  std::cout << "m_i0 " << m_i0 << "\n";
  std::cout << "m_i1 " << m_i1 << "\n";
  std::cout << "m_l " << m_l << "\n";
  std::cout << "m_n " << m_n << "\n";
  std::cout << "m_s " << m_s << "\n";
  std::cout << "m_name " << m_name << "\n";
  std::cout << "m_data.m_l " << m_data.m_l << "\n";
  std::cout << "m_data.m_ir " << m_data.m_ir << "\n";
  std::cout << "m_data.m_angle " << m_data.m_angle << "\n";
  std::cout << "R = m_data.m_l/m_data.m_angle " << m_data.m_l/m_data.m_angle << "\n";
  std::cout << "m_data.m_atw00 " << m_data.m_atw00 << "\n";
  std::cout << "m_data.m_atw01 " << m_data.m_atw01 << "\n";
  std::cout << "m_data.m_btw00 " << m_data.m_btw00 << "\n";
  std::cout << "m_data.m_btw01 " << m_data.m_btw01 << "\n";

//teapot->survey(surveyData,m_i0,m_i0+1);
/*
  double xLS = surveyData.survey().x();
  double yLS = surveyData.survey().y();
  double zLS = surveyData.survey().z();
  std::cout << "xLS      " << xLS      << " yLS      " << yLS      << " zLS      " << zLS      << "\n";
*/
std::cout << "       xS[m_i0] " << ETEAPOT::DipoleTracker::xS[m_i0] << " member yS[m_i0] " << ETEAPOT::DipoleTracker::yS[m_i0] << " member zS[m_i0] " << ETEAPOT::DipoleTracker::zS[m_i0] << " member nS[m_i0] " << ETEAPOT::DipoleTracker::nS[m_i0] << "\n";
PacSurvey survey=m_data.m_slices[0].survey();
std::cout << "member xS[m_i0] " << survey.x() << " member yS[m_i0] " << survey.y() << " member zS[m_i0] " << survey.z() << "\n";

//                                            "Everything" about design/central orbit
//double oldT = ba.getElapsedTime();
  double e0 = ba.getEnergy();
  double m0 = ba.getMass();
  double q0 = ba.getCharge();
  double t0 = ba.getElapsedTime();
                                // RevFreq
                                // Macrosize
                                // G
  double L0 = ba.getL();
//                                            "Everything" about design/central orbit

  double oldT = t0;
         q0   = UAL::elemCharge;
  double p0   = sqrt(e0*e0 - m0*m0);

std::cout << "double q0 = ba.getCharge() = " << q0 << "\n";
std::cout << "UAL::elemCharge = " << UAL::elemCharge << "\n";
//               m_time
//               m_revfreq
//               m_macrosize
//               m_G
