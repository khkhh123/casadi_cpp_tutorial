#include <casadi/casadi.hpp>
#include <iostream>
#include <vector>

using namespace casadi;

int main() {
    // 1. 파라미터 설정
    int N = 20;           // Prediction Horizon (예측 지평)
    double dt = 0.1;      // Time step
    double m = 1500.0;    // 질량 (kg)
    double v_ref = 100.0/3.6;  // 목표 속도 (m/s)

    // 2. 심볼릭 변수 정의 (동역학 모델을 위한 변수)
    SX v = SX::sym("v");  // 상태 변수: 속도
    SX u = SX::sym("u");  // 제어 입력: 힘(토크 대응)

    // 3. 동역학 모델 정의 (v_dot = f(v, u))
    SX v_dot = u / m;
    Function f = Function("f", {v, u}, {v_dot});

    // 4. 최적화 문제 구성 (NLP)
    SX J = 0;             // Cost function
    std::vector<SX> g;    // Constraints (제약 조건)
    
    // 설계 변수 (Decision Variables) 벡터
    SX X = SX::sym("X", N + 1); // 미래 속도 경로
    SX U = SX::sym("U", N);     // 미래 제어 입력

    // 초기 상태 제약 조건을 위한 파라미터
    SX v0 = SX::sym("v0");

    g.push_back(X(0) - v0); // 초기 속도 고정
    
    for (int k = 0; k < N; ++k) {
        // 비용 함수: (현재속도 - 목표속도)^2 + 제어입력 페널티
        J += pow(X(k) - v_ref, 2) + 1e-7 * pow(U(k), 2);

        // 동역학 제약 조건 (Euler Integration: x_next = x + f*dt)
        SX x_next = X(k) + f(std::vector<SX>{X(k), U(k)})[0] * dt;
        g.push_back(X(k + 1) - x_next);
    }

    // 5. NLP 솔버 생성
    SX dict_X = SX::vertcat({X, U});
    SX dict_G = SX::vertcat(g);
    SXDict nlp = {{"x", dict_X}, {"f", J}, {"g", dict_G}, {"p", v0}};

    Dict opts;
    opts["qpsol"] = "qrqp";        // 내장된 QP 솔버 사용
    opts["print_header"] = false;
    opts["print_iteration"] = false;
    opts["print_time"] = false;
    opts["print_status"] = false;
    opts["qpsol_options.print_iter"] = false;

    Function solver = nlpsol("solver", "sqpmethod", nlp, opts);

    // 6. 시뮬레이션 루프
    double current_v = 0.0; // 정지 상태에서 시작
    std::vector<double> x_init(N + 1 + N, 0.0); // 초기 추측값

    std::cout << "Time | Velocity | Control Input" << std::endl;
    for (int i = 0; i < 200; ++i) {
        // 제약 조건 경계 설정 (Bounds)
        std::vector<double> lbg(N + 1, 0.0), ubg(N + 1, 0.0); // Equality constraints = 0
        
        // 솔버 호출
        DMDict arg;
        arg["x0"] = x_init;
        arg["p"] = current_v; // 현재 속도를 파라미터로 전달
        arg["lbg"] = 0; 
        arg["ubg"] = 0;
        
        // 제어 입력 범위 제한 (예: -3000N ~ 3000N)
        std::vector<double> lbx(N + 1 + N, -std::numeric_limits<double>::infinity());
        std::vector<double> ubx(N + 1 + N, std::numeric_limits<double>::infinity());
        for(int j=N+1; j<N+1+N; ++j) {
            lbx[j] = -3000.0;
            ubx[j] = 3000.0;
        }
        arg["lbx"] = lbx;
        arg["ubx"] = ubx;

        auto res = solver(arg);
        
        // 결과 추출
        std::vector<double> sol_x = std::vector<double>(res.at("x"));
        double control_u = sol_x[N + 1]; // 첫 번째 제어 입력 적용

        // 실제 시스템 업데이트 (단순화된 물리 엔진)
        current_v += (control_u / m) * dt;

        printf("%3.1f s | %6.2f km/h | %7.1f N\n", i * dt, current_v * 3.6, control_u);
    }

    return 0;
}